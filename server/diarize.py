r"""Диаризация разговора: WAV → «кто что сказал» (markdown).

Конвейер: pyannote (спикер-интервалы, GPU) + whisper-server :8080 (текст с
таймкодами) → мерж по перекрытию → метка «Дима» по эталону голоса.

Запуск (на home-pc, venv-diar):
    venv-diar\Scripts\python.exe diarize.py audio\2026-07-12_213012.wav
    ... --all            # все необработанные WAV из audio/
    ... --ref voice.wav  # свой эталон (по умолчанию base/voice_profile.wav)

Требует в .env: HF_TOKEN (read-токен HuggingFace, условия моделей приняты:
pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0).
Результат: <имя>.speakers.md рядом с WAV + копия в base/INBOX/ (разберёт ночной
разбор) и immutable <имя>.speakers.json с точными sample-границами.
Повторный прогон файла с готовым .speakers.md пропускается.
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from hallucinations import is_hallucination
from memory_epoch import load_memory_layout
from pomnit_transcript_sidecar import inspect_wav, load_for_wav, sidecar_path, write_for_wav

load_dotenv(ROOT / ".env")
MEMORY_LAYOUT = load_memory_layout(ROOT)

# STT_URL — общий переключатель STT (whisper :8080 / gigaam :8081), как в мосте.
WHISPER_URL = (os.environ.get("STT_URL") or os.environ.get("WHISPER_URL")
               or "http://127.0.0.1:8080/inference")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO_DIR = Path(os.environ.get("AUDIO_BRIDGE_DIR", ROOT / "audio"))
REF_DEFAULT = MEMORY_LAYOUT.owner_assets / "voice_profile.wav"
INBOX = MEMORY_LAYOUT.owner_base / "INBOX"
VOICES = MEMORY_LAYOUT.owner_assets / "voices"  # operational, not part of a memory epoch
EMB_DIR = VOICES / "embeddings"            # отпечатки кластеров каждого разговора
SIM_THRESHOLD = 0.4  # косинусная близость к эталону, ниже — «не Дима»


def transcribe(path: Path) -> list[dict]:
    """whisper-server → сегменты [{text,start,end}]. multipart руками, без requests."""
    boundary = "----diarize-boundary"
    body = io.BytesIO()

    def field(name, value):
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    field("response_format", "verbose_json")
    field("temperature", "0.0")
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"{path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    body.write(path.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(WHISPER_URL, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    out = []
    prev = None
    for s in data.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text or is_hallucination(text):
            continue
        if text == prev:
            continue  # зацикливание модели на тишине
        prev = text
        out.append({"text": text, "start": float(s.get("start", 0)), "end": float(s.get("end", 0))})
    return out


def load_audio(path: Path) -> dict:
    """WAV PCM16 → {'waveform','sample_rate'} — обход torchcodec (не заводится
    на Windows без FFmpeg-DLL); наши файлы всегда простой PCM16 моно."""
    import numpy as np
    import torch
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    wf = torch.from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0)
    return {"waveform": wf, "sample_rate": sr}


@dataclass(frozen=True)
class _SttPcmWindow:
    """Exact source samples copied into the temporary WAV sent to STT."""

    sample_rate: int
    start_sample: int
    end_sample: int
    frames: bytes


def _stt_pcm_window(path: Path, start: float, end: float) -> _SttPcmWindow:
    """Read exactly the clamped PCM window used by ``transcribe_span``.

    ``wave.readframes`` may return fewer frames at EOF.  Deriving ``end_sample``
    from the bytes actually read makes the sidecar evidence range match the STT
    input rather than the unclamped pyannote turn.
    """
    with wave.open(str(path)) as source:
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        frame_bytes = source.getnchannels() * source.getsampwidth()
        start_sample = min(frame_count, int(max(0.0, start) * sample_rate))
        requested_frames = int(max(0.2, end - start) * sample_rate)
        source.setpos(start_sample)
        frames = source.readframes(requested_frames)
    if frame_bytes <= 0 or len(frames) % frame_bytes:
        raise ValueError("WAV returned an incomplete PCM frame")
    actual_frames = len(frames) // frame_bytes
    return _SttPcmWindow(
        sample_rate=sample_rate,
        start_sample=start_sample,
        end_sample=start_sample + actual_frames,
        frames=frames,
    )


def transcribe_span(
    path: Path,
    start: float,
    end: float,
    *,
    pcm_window: _SttPcmWindow | None = None,
) -> str:
    """Расшифровка ОДНОГО интервала (реплики по pyannote). Целый длинный файл
    whisper.cpp не переваривает: после минут шума уходит в петлю галлюцинаций
    (грабля 13.07), а на коротких кусках работает отлично."""
    window = pcm_window or _stt_pcm_window(path, start, end)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(window.sample_rate)
        o.writeframes(window.frames)
    boundary = "----diarize-boundary"
    body = io.BytesIO()
    for name, value in (("response_format", "json"), ("temperature", "0.0")):
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"span.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    body.write(buf.getvalue())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(WHISPER_URL, data=body.getvalue(), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return " ".join((data.get("text") or "").split())


def merge_turns(turns, max_gap: float = 1.2):
    """Соседние реплики одного спикера с паузой < max_gap склеиваются."""
    spans = []
    for start, end, spk in sorted(turns):
        if spans and spans[-1][2] == spk and start - spans[-1][1] <= max_gap:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end, spk])
    return spans


_PIPE = None
_EMB = None


def _pipeline():
    """Пайплайн диаризации, один на процесс (--all гоняет десятки файлов)."""
    global _PIPE
    if _PIPE is None:
        import torch
        from pyannote.audio import Pipeline
        _PIPE = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")  # токен из env
        if torch.cuda.is_available():
            _PIPE.to(torch.device("cuda"))
    return _PIPE


def _embedder():
    """Модель эмбеддингов голоса, одна на процесс."""
    global _EMB
    if _EMB is None:
        import torch
        from pyannote.audio import Inference, Model
        _EMB = Inference(Model.from_pretrained("pyannote/embedding"), window="whole")
        if torch.cuda.is_available():
            _EMB.to(torch.device("cuda"))
    return _EMB


MERGE_SIM = float(os.environ.get("DIAR_MERGE_SIM", 0.55))  # косинус: выше — один голос


def diarize(path: Path):
    """pyannote → [(start, end, 'SPEAKER_00'), ...]"""
    out = _pipeline()(load_audio(path))
    # pyannote 4.x возвращает обёртку, 3.x — сразу Annotation; на записи без речи
    # обёртка может быть пустой (атрибуты None, itertracks нет) — это не крэш
    ann = getattr(out, "speaker_diarization", None) or getattr(out, "diarization", None) or out
    if not hasattr(ann, "itertracks"):
        return []
    return [(turn.start, turn.end, spk) for turn, _, spk in ann.itertracks(yield_label=True)]


def _ref_embedding(ref: Path):
    """Нормированный эмбеддинг эталона голоса Димы (или None)."""
    if not (ref.exists() and HF_TOKEN):
        return None
    try:
        import numpy as np
        v = _embedder()(load_audio(ref))
        return v / np.linalg.norm(v)
    except Exception as e:
        print(f"  эталон не прочитан: {e}", flush=True)
        return None


def _voice_library(ref: Path) -> dict:
    """Подтверждённые голоса: эталон Димы + только owner_manual JSON.

    Старые профили, которые voice_learn создавал из контекстных догадок,
    остаются на диске для аудита, но не могут приписать чужую реплику человеку.
    """
    import numpy as np
    lib = {}
    dima = _ref_embedding(ref)
    if dima is not None:
        lib["Дима"] = dima
    for f in sorted(VOICES.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if ("name" not in d or "embedding" not in d or
                    d.get("confirmation") != "owner_manual"):
                continue  # служебные файлы (learn_state.json и пр.) — не голоса
            v = np.array(d["embedding"], dtype=float)
            lib[d["name"]] = v / np.linalg.norm(v)
        except Exception as e:
            print(f"  голос {f.name} не прочитан: {e}", flush=True)
    return lib


def identify_speakers(embs: dict, library: dict) -> dict:
    """Кластеры → имена по всей библиотеке голосов: жадно по убыванию близости,
    каждый эталон и каждый кластер используются один раз, порог SIM_THRESHOLD."""
    import numpy as np
    pairs = sorted(
        ((float(np.dot(ref, v)), name, spk)
         for name, ref in library.items() for spk, v in embs.items()),
        reverse=True)
    named, used = {}, set()
    for sim, name, spk in pairs:
        if sim <= SIM_THRESHOLD:
            break
        if name in used or spk in named:
            continue
        print(f"  {spk} = {name} (близость {sim:.2f})", flush=True)
        named[spk] = name
        used.add(name)
    return named


def merge_speakers(path: Path, turns, embs=None, protected=None):
    """Схлопывает ложные кластеры: в шумном многолюдном помещении pyannote дробит
    один голос на несколько SPEAKER. Сливает пары с косинусной близостью > MERGE_SIM
    (по связности). protected (кластер Димы) НЕ сливается ни с кем — метка «Дима»
    защищена от растворения агрессивной склейкой."""
    import numpy as np
    if embs is None:
        embs = speaker_embeddings(path, turns)
    spks = list(embs)
    parent = {s: s for s in spks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    protected = protected or set()
    if isinstance(protected, str):
        protected = {protected}
    for i in range(len(spks)):
        for j in range(i + 1, len(spks)):
            a, b = spks[i], spks[j]
            if a in protected or b in protected:
                continue  # опознанных (Дима и выученные голоса) ни с кем не сливаем
            if float(np.dot(embs[a], embs[b])) > MERGE_SIM:
                parent[find(a)] = find(b)
    remap = {s: find(s) for s in spks}
    merged = [(a, b, remap.get(spk, spk)) for a, b, spk in turns]
    before, after = len({s for _, _, s in turns}), len(set(remap.values()))
    if before != after:
        print(f"  склейка спикеров: {before} -> {after}", flush=True)
    return merged


def speaker_embeddings(path: Path, turns) -> dict:
    """Средний эмбеддинг каждого спикера по его самым длинным репликам (до 60 с)."""
    import numpy as np
    import torch
    from pyannote.audio import Inference
    from pyannote.core import Segment

    inf = _embedder()
    with wave.open(str(path)) as w:
        dur = w.getnframes() / w.getframerate()
    by_spk: dict[str, list] = {}
    for start, end, spk in turns:
        by_spk.setdefault(spk, []).append((end - start, start, end))
    result = {}
    for spk, segs in by_spk.items():
        segs.sort(reverse=True)
        embs, used = [], 0.0
        for length, start, end in segs:
            if length < 1.0 or used > 60:
                break
            end = min(end, dur - 0.05)  # строго меньше длительности: crop падает на t==dur
            if end - start < 1.0:
                continue
            embs.append(inf.crop(load_audio(path), Segment(start, end)))
            used += length
        if embs:
            v = np.mean(np.vstack(embs), axis=0)
            result[spk] = v / np.linalg.norm(v)
    return result


def label_speakers(turns, named: dict) -> dict:
    """SPEAKER_XX → имя: опознанные — из библиотеки, остальные «Голос N»."""
    names, n = {}, 0
    for _, _, spk in turns:
        if spk in names:
            continue
        if spk in named:
            names[spk] = named[spk]
        else:
            n += 1
            names[spk] = f"Голос {n}"
    return names


def export_embeddings(stem: str, turns, embs: dict, names: dict) -> None:
    """Отпечатки голосов разговора → base/voices/embeddings/<stem>.json —
    сырьё для voice_learn.py (обучение новых голосов по контексту). Ключи —
    итоговые лейблы («Дима», «Голос 2»…), эмбеддинги слитых кластеров усредняются."""
    import numpy as np
    by_label: dict[str, list] = {}
    speech: dict[str, float] = {}
    for start, end, spk in turns:
        label = names.get(spk)
        if label is None or spk not in embs:
            continue
        by_label.setdefault(label, []).append(embs[spk])
        speech[label] = speech.get(label, 0.0) + (end - start)
    out = {}
    for label, vecs in by_label.items():
        v = np.mean(np.vstack(vecs), axis=0)
        v = v / np.linalg.norm(v)
        out[label] = {"embedding": [round(float(x), 5) for x in v],
                      "speech_sec": round(speech.get(label, 0.0), 1)}
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    (EMB_DIR / f"{stem}.json").write_text(
        json.dumps({"file": stem, "clusters": out}, ensure_ascii=False), encoding="utf-8")


def _norm_echo(s: str) -> str:
    return re.sub(r"[^а-яёa-z0-9]+", "", s.lower())


def _dedupe_echo(pairs: list) -> list:
    """R3-4: «эхо» на границах реплик — конец фразы спикера A повторяется в начале
    реплики B (перекрытие окон распознавания). По аудиту 8-12% реплик. Если
    нормализованный хвост A (≥15 знаков) совпадает с началом B — режем повтор из B."""
    out = []
    for spk, text in pairs:
        if out:
            prev = _norm_echo(out[-1][1][-120:])
            words = text.split()
            # ищем самый длинный префикс B (по словам), который уже прозвучал у A
            cut = 0
            for n in range(len(words), 0, -1):
                head = _norm_echo(" ".join(words[:n]))
                if len(head) >= 15 and prev.endswith(head):
                    cut = n
                    break
            if cut:
                text = " ".join(words[cut:]).strip()
        if not text:
            continue
        if out and out[-1][0] == spk:
            out[-1] = (spk, out[-1][1] + " " + text)
        else:
            out.append((spk, text))
    return out


def _legacy_markdown(stamp: str, participants: list[str], turns: list[dict],
                     empty_label: str | None = None) -> str:
    if not turns:
        return f"# Разговор {stamp}: {empty_label or 'речи не найдено'}\n"
    md = "\n\n".join(
        f"**{turn['speaker_label']}:** {turn['text']}" for turn in turns)
    header = (f"# Разговор {stamp} (диаризация)\n\n"
              f"Участники: {', '.join(participants)}\n\n")
    return header + md + "\n"


def _publish_legacy_markdown(path: Path, text: str) -> None:
    """Preserve the existing Markdown output after the sidecar is durable."""
    out = path.with_suffix(".speakers.md")
    out.write_text(text, encoding="utf-8")
    if INBOX.exists():
        (INBOX / f"{path.stem}_diarized.md").write_text(text, encoding="utf-8")
        mark_webhook_duplicates(path.stem)


# ПОЧЕМУ ЗДЕСЬ НЕТ РАЗРЕЗА СЛИПШИХСЯ РЕПЛИК: process() распознаёт КАЖДЫЙ
# отрезок диаризации отдельно (transcribe_span), поэтому текст двух людей
# в один кусок не попадает. Реальная причина «слипшихся» абзацев — расширение
# границ отрезка (start-0.25 / end+0.4): в кусок попадает хвост соседа. Лечит
# это _dedupe_echo, который применяется к готовым репликам ниже (07.08).

def process(path: Path, ref: Path) -> bool:
    out = path.with_suffix(".speakers.md")
    exact = sidecar_path(path)
    if out.exists():
        # A new-format output must still prove that its immutable sidecar names
        # these exact WAV bytes.  Old Markdown without a sidecar is deliberately
        # left alone and is never retro-imported.
        if exact.exists():
            load_for_wav(exact, path)
        return False
    if exact.exists():
        # Crash recovery: sidecar publication precedes legacy Markdown.  Rebuild
        # the latter without another STT/model pass and without changing JSON.
        payload = load_for_wav(exact, path)
        turns = payload["turns"]
        participants = list(dict.fromkeys(
            turn["speaker_label"] for turn in turns))
        empty = {
            "no_speech": "речи не найдено",
            "no_clear_speech": "внятной речи не найдено",
        }.get(payload["coverage"]["status"])
        _publish_legacy_markdown(
            path, _legacy_markdown(path.stem, participants, turns, empty))
        return bool(turns)
    source_fingerprint = inspect_wav(path)
    print(f"диаризую {path.name} …")
    turns = diarize(path)
    if not turns:
        print("  спикеры не найдены")
        # маркер обязателен: без него diar_job берёт файл каждый тик ВЕЧНО
        # (застревание 13-19.07: очередь стояла на одном шумовом файле)
        write_for_wav(
            path, [], coverage_status="no_speech",
            expected_source=source_fingerprint)
        _publish_legacy_markdown(
            path, _legacy_markdown(path.stem, [], [], "речи не найдено"))
        # R1-8: тихие потери — в журнал, чтобы их было видно, а не «файл просто исчез»
        import datetime as _dt
        with (VOICES / "no_speech.log").open("a", encoding="utf-8") as log:
            log.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {path.name}\n")
        return False
    embs = speaker_embeddings(path, turns)        # эмбеддинги — один раз
    named = identify_speakers(embs, _voice_library(ref))  # вся библиотека, ДО склейки
    turns = merge_speakers(path, turns, embs=embs, protected=set(named))
    names = label_speakers(turns, named)
    export_embeddings(path.stem, turns, embs, names)  # сырьё для voice_learn
    evidence_turns = []
    partial = False
    for start, end, spk in merge_turns(turns):
        if end - start < 1.0:
            partial = True
            continue
        try:
            pcm_window = _stt_pcm_window(path, start - 0.25, end + 0.4)
            if pcm_window.end_sample <= pcm_window.start_sample:
                partial = True
                continue
            text = transcribe_span(
                path, start - 0.25, end + 0.4, pcm_window=pcm_window)
        except Exception as e:
            print(f"  span {start:.0f}s: ошибка whisper: {e}")
            partial = True
            continue
        if not text or is_hallucination(text):
            partial = True
            continue
        name = names.get(spk, "Голос ?")
        # Evidence stays 1:1 with an actual STT request.  Human Markdown may
        # merge adjacent labels below, but the canonical quote never claims a
        # bounding interval that included audio not sent in that request.
        evidence_turns.append({
            "speaker_label": name, "text": text,
            "start_sample": pcm_window.start_sample,
            "end_sample": pcm_window.end_sample,
        })
    if not evidence_turns:
        print("  внятной речи не найдено")
        write_for_wav(
            path, [], coverage_status="partial" if partial else "no_clear_speech",
            expected_source=source_fingerprint)
        _publish_legacy_markdown(
            path, _legacy_markdown(
                path.stem, [], [], "внятной речи не найдено"))
        return False
    # эхо на стыках: хвост фразы соседа попадает в расширенный отрезок и звучит
    # дважды — «Ага, но он его не вставил, да?» и у Димы, и у второго носителя (аудит 06.08)
    markdown_pairs = _dedupe_echo([
        (turn["speaker_label"], turn["text"]) for turn in evidence_turns])
    if (len(markdown_pairs) != len(evidence_turns)
            or any(pair != (turn["speaker_label"], turn["text"])
                   for pair, turn in zip(markdown_pairs, evidence_turns))):
        print(f"  эхо на стыках подчищено: было {len(evidence_turns)} реплик, "
              f"стало {len(markdown_pairs)}")
    markdown_turns = [
        {"speaker_label": speaker, "text": text}
        for speaker, text in markdown_pairs
    ]
    stamp = path.stem
    write_for_wav(
        path, evidence_turns,
        coverage_status="partial" if partial else "complete",
        expected_source=source_fingerprint)
    _publish_legacy_markdown(
        path, _legacy_markdown(
            stamp, list(dict.fromkeys(names.values())), markdown_turns))
    print(f"  готово: {out.name}, спикеров {len(set(names.values()))}, "
          f"реплик {len(evidence_turns)}")
    return True


def mark_webhook_duplicates(stamp: str) -> None:
    """Дедуп карточек (A5): тот же разговор приходит и legacy-вебхуком (его STT хуже),
    и нашей диаризацией. Диаризованная версия главнее: вебхук-файлы из временного
    окна записи помечаются разобранными — ночной разбор видит разговор один раз.
    Нет диаризации — вебхук остаётся резервом и разбирается как раньше."""
    from datetime import datetime, timedelta
    try:
        # Synced NAND files carry an idempotency suffix after the timestamp;
        # use the same capture window for webhook dedup as ordinary live WAVs.
        start = datetime.strptime(stamp[:17], "%Y-%m-%d_%H%M%S")
        with wave.open(str(AUDIO_DIR / f"{stamp}.wav")) as w:
            dur = w.getnframes() / w.getframerate()
    except Exception:
        return
    window_end = start + timedelta(seconds=dur) + timedelta(minutes=40)
    for f in INBOX.glob("*.md"):
        if f.stem.endswith("_diarized"):
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2}_\d{6})_", f.name)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(1), "%Y-%m-%d_%H%M%S")
        except ValueError:
            continue
        if not (start <= t <= window_end):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if "_разобрано" in text:
                continue
            with f.open("a", encoding="utf-8") as fh:
                fh.write(f"\n_разобрано: дубль, есть {stamp}_diarized_\n")
            print(f"  дубль вебхука помечен: {f.name}", flush=True)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", help="WAV-файл")
    ap.add_argument("--all", action="store_true", help="все необработанные из audio/")
    ap.add_argument("--ref", default=str(REF_DEFAULT), help="эталон голоса Димы")
    # мультибазы (C5): у второго носителя своя база — расшифровка должна лечь
    # в его INBOX, а не в дневник Димы
    ap.add_argument("--inbox", default=None, help="куда класть копию расшифровки")
    # своя библиотека голосов на носителя: иначе люди из окружения Димы
    # начнут «узнаваться» в чужих разговорах — это утечка знания о нём
    ap.add_argument("--voices", default=None, help="библиотека голосов носителя")
    args = ap.parse_args()
    if args.inbox:
        global INBOX
        INBOX = Path(args.inbox)
        INBOX.mkdir(parents=True, exist_ok=True)
    if args.voices:
        global VOICES, EMB_DIR
        VOICES = Path(args.voices)
        EMB_DIR = VOICES / "embeddings"
        VOICES.mkdir(parents=True, exist_ok=True)
        EMB_DIR.mkdir(parents=True, exist_ok=True)
    if not HF_TOKEN:
        sys.exit("HF_TOKEN не задан в .env — см. handover/RUNBOOK.md, «Диаризация»")
    ref = Path(args.ref)
    if args.all:
        import time
        done = 0
        for wav in sorted(AUDIO_DIR.glob("*.wav")):
            if time.time() - wav.stat().st_mtime < 600:
                print(f"пропускаю {wav.name}: ещё пишется")
                continue
            done += process(wav, ref)
        print(f"обработано: {done}")
    elif args.wav:
        process(Path(args.wav), ref)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
