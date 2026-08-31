r"""GigaAM-сервер (:8081) — STT русской речи, альтернатива whisper-server.

Модель: GigaAM v3 (Сбер, https://github.com/salute-developers/GigaAM), SOTA на
русском, MIT. По умолчанию v3_e2e_rnnt — с пунктуацией и нормализацией.

API повторяет whisper-server ровно настолько, насколько им пользуются наши
клиенты (мост audio_bridge_core.py и diarize.py):
  POST /inference  multipart, поле file (WAV) → {"text", "segments":[{text,start,end}]}
  GET  /           строка статуса (для curl-проверок и сторожей)
Лишние поля формы (response_format, temperature, prompt) принимаются и
игнорируются — у GigaAM нет промпта.

Переключение конвейера: STT_URL=http://127.0.0.1:8081/inference в .env →
перезапуск службы моста (и службы ассистента, чтобы ночная диаризация получила
тот же URL). Откат: убрать STT_URL, перезапустить те же службы.

Запуск (venv с gigaam): python gigaam_server.py
Ограничение transcribe — 25 с; длиннее режем на окна по 24 с (longform-режим
GigaAM требует torchcodec, который на Windows не заводится).
"""
import asyncio
import io
import os
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

# gigaam.load_audio зовёт ffmpeg из PATH; в PATH службы Windows его нет —
# добавляем каталог из FFMPEG_DIR (задаётся при установке службы) в начало PATH.
_FFMPEG_DIR = os.environ.get("FFMPEG_DIR", "")
if _FFMPEG_DIR:
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

PORT = int(os.environ.get("GIGAAM_PORT", 8081))
MODEL_NAME = os.environ.get("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE = os.environ.get("GIGAAM_DEVICE", "")          # пусто = cuda, если доступна
CACHE = os.environ.get("GIGAAM_CACHE", "")            # пусто = ~/.cache/gigaam
WINDOW_SEC = 24                                       # < лимита transcribe (25 с)

_EXECUTOR = ThreadPoolExecutor(max_workers=1)         # GPU-работа строго по одной
_MODEL = None


def _load_model():
    global _MODEL
    import gigaam
    import torch
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = gigaam.load_model(MODEL_NAME, device=device,
                               download_root=(CACHE or None))
    print(f"gigaam: модель {MODEL_NAME} загружена на {device}", flush=True)


def _split_wav(data: bytes) -> list[tuple[bytes, float, float]]:
    """WAV → [(wav_окна, start_sec, end_sec)]. Целиком, если короче WINDOW_SEC."""
    with wave.open(io.BytesIO(data)) as w:
        sr, sw, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        total = w.getnframes()
        out = []
        pos = 0
        step = WINDOW_SEC * sr
        min_frames = sr // 2  # окна короче 0.5 с роняют stft модели (n_fft=320)
        while pos < total:
            n = min(step, total - pos)
            frames = w.readframes(n)
            if n < min_frames:
                pos += n
                continue
            buf = io.BytesIO()
            with wave.open(buf, "wb") as o:
                o.setnchannels(ch)
                o.setsampwidth(sw)
                o.setframerate(sr)
                o.writeframes(frames)
            out.append((buf.getvalue(), pos / sr, min(pos + step, total) / sr))
            pos += step
    return out


def _transcribe_blocking(data: bytes) -> dict:
    segments = []
    for wav_bytes, start, end in _split_wav(data):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav_bytes)
            tmp.close()
            # одно битое окно не должно ронять весь файл — раньше исключение
            # уходило 500-кой и терялся ЦЕЛИКОМ кусок дня; теперь окно пропускается
            try:
                text = str(_MODEL.transcribe(tmp.name)).strip()
            except Exception as e:
                print(f"gigaam: окно {start:.0f}-{end:.0f}с пропущено: {e}", flush=True)
                continue
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        if text:
            segments.append({"text": text, "start": round(start, 2), "end": round(end, 2)})
    return {"text": " ".join(s["text"] for s in segments), "segments": segments,
            "language": "ru"}


async def handle_inference(request: web.Request) -> web.Response:
    form = await request.post()
    field = form.get("file")
    if field is None or not hasattr(field, "file"):
        return web.json_response({"error": "no file field"}, status=400)
    data = field.file.read()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, _transcribe_blocking, data)
    except Exception as e:
        print(f"gigaam: ошибка расшифровки: {e}", flush=True)
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(result)


async def handle_ping(request: web.Request) -> web.Response:
    return web.Response(text=f"gigaam-server alive (model={MODEL_NAME})")


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/inference", handle_inference)
app.router.add_get("/", handle_ping)

if __name__ == "__main__":
    _load_model()  # падаем сразу, если модель не встала — менеджер служб перезапустит
    print(f"gigaam-server: слушаю 127.0.0.1:{PORT}", flush=True)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
