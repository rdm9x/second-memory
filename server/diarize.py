r"""Диаризация разговора: WAV → «кто что сказал» (markdown).

Конвейер: pyannote (спикер-интервалы, GPU) + STT-сервер :8080 (текст с
таймкодами) → мерж по перекрытию → метка «Владелец» по эталону голоса.
Голоса собеседников копятся накопительно: каждый разговор оставляет отпечатки
кластеров, ночное обучение подтверждает имена по контексту —
система со временем узнаёт окружение владельца по голосу.

Запуск (в venv диаризации):
    python diarize.py audio\2026-07-12_213012.wav
    ... --all            # все необработанные WAV из audio/
    ... --ref voice.wav  # свой эталон (по умолчанию base/voice_profile.wav)

Требует в .env: HF_TOKEN (read-токен HuggingFace, условия моделей приняты:
pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0).
Результат: <имя>.speakers.md рядом с WAV + копия в base/INBOX/ (разберёт ночной
разбор). Повторный прогон файла с готовым .speakers.md пропускается.
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from hallucinations import is_hallucination

load_dotenv(ROOT / ".env")

# STT_URL — общий переключатель STT (whisper :8080 / gigaam :8081), как в мосте.
WHISPER_URL = (os.environ.get("STT_URL") or os.environ.get("WHISPER_URL")
               or "http://127.0.0.1:8080/inference")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO_DIR = Path(os.environ.get("AUDIO_BRIDGE_DIR", ROOT / "audio"))
REF_DEFAULT = ROOT / "base" / "voice_profile.wav"
INBOX = ROOT / "base" / "INBOX"
VOICES = ROOT / "base" / "voices"          # выученные голоса: <имя>.json (voice_learn)
EMB_DIR = VOICES / "embeddings"            # отпечатки кластеров каждого разговора
SIM_THRESHOLD = 0.4  # косинусная близость к эталону, ниже — «не владелец»

OWNER_LABEL = "Владелец"   # метка хозяина дневника в расшифровках


def transcribe(path: Path) -> list[dict]:
    """STT-сервер → сегменты [{text,start,end}]. multipart руками, без requests."""
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


def transcribe_span(path: Path, start: float, end: float) -> str:
    """Расшифровка ОДНОГО интервала (реплики по pyannote). Целый длинный файл
    whisper.cpp не переваривает: после минут шума уходит в петлю галлюцинаций,
    а на коротких кусках работает отлично."""
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        w.setpos(int(max(0.0, start) * sr))
        frames = w.readframes(int(max(0.2, end - start) * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(sr)
        o.writeframes(frames)
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
    """Нормированный эмбеддинг эталона голоса владельца (или None)."""
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
    """Все известные голоса: владелец (wav-эталон) + выученные из base/voices/*.json
    (их копит voice_learn по подтверждениям из контекста разговоров)."""
    import numpy as np
    lib = {}
    owner = _ref_embedding(ref)
    if owner is not None:
        lib[OWNER_LABEL] = owner
    for f in sorted(VOICES.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if "name" not in d or "embedding" not in d:
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
    (по связности). protected (кластер владельца) НЕ сливается ни с кем — его
    метка защищена от растворения агрессивной склейкой."""
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
                continue  # опознанные (владелец и выученные голоса) ни с кем не сливаем
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
    сырьё для voice_learn (обучение новых голосов по контексту). Ключи —
    итоговые лейблы («Владелец», «Голос 2»…), эмбеддинги слитых кластеров
    усредняются."""
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
    """«Эхо» на границах реплик — конец фразы спикера A повторяется в начале
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


# ПОЧЕМУ ЗДЕСЬ НЕТ РАЗРЕЗА СЛИПШИХСЯ РЕПЛИК: process() распознаёт КАЖДЫЙ
# отрезок диаризации отдельно (transcribe_span), поэтому текст двух людей
# в один кусок не попадает. Реальная причина «слипшихся» абзацев — расширение
# границ отрезка (start-0.25 / end+0.4): в кусок попадает хвост соседа. Лечит
# это _dedupe_echo, который применяется к готовым репликам ниже.

def process(path: Path, ref: Path) -> bool:
    out = path.with_suffix(".speakers.md")
    if out.exists():
        return False
    print(f"диаризую {path.name} …")
    turns = diarize(path)
    if not turns:
        print("  спикеры не найдены")
        # маркер обязателен: без него ночной конвейер берёт файл каждый тик
        # ВЕЧНО (очередь может застрять на одном шумовом файле)
        out.write_text(f"# Разговор {path.stem}: речи не найдено\n", encoding="utf-8")
        # тихие потери — в журнал, чтобы их было видно, а не «файл просто исчез»
        import datetime as _dt
        with (VOICES / "no_speech.log").open("a", encoding="utf-8") as log:
            log.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {path.name}\n")
        return False
    embs = speaker_embeddings(path, turns)        # эмбеддинги — один раз
    named = identify_speakers(embs, _voice_library(ref))  # вся библиотека, ДО склейки
    turns = merge_speakers(path, turns, embs=embs, protected=set(named))
    names = label_speakers(turns, named)
    export_embeddings(path.stem, turns, embs, names)  # сырьё для voice_learn
    lines = []
    for start, end, spk in merge_turns(turns):
        if end - start < 1.0:
            continue
        try:
            text = transcribe_span(path, start - 0.25, end + 0.4)
        except Exception as e:
            print(f"  span {start:.0f}s: ошибка STT: {e}")
            continue
        if not text or is_hallucination(text):
            continue
        name = names.get(spk, "Голос ?")
        if lines and lines[-1][0] == name:
            lines[-1][1].append(text)
        else:
            lines.append([name, [text]])
    if not lines:
        print("  внятной речи не найдено")
        out.write_text(f"# Разговор {path.stem}: внятной речи не найдено\n", encoding="utf-8")
        return False
    # эхо на стыках: хвост фразы соседа попадает в расширенный отрезок и звучит
    # дважды — одна и та же фраза у обоих участников (поймано аудитом расшифровок)
    pairs = [(n, " ".join(parts)) for n, parts in lines]
    clean = _dedupe_echo(pairs)
    if len(clean) != len(pairs) or any(a[1] != b[1] for a, b in zip(pairs, clean)):
        print(f"  эхо на стыках подчищено: было {len(pairs)} реплик, стало {len(clean)}")
    md = "\n\n".join(f"**{n}:** {t}" for n, t in clean)
    stamp = path.stem
    header = f"# Разговор {stamp} (диаризация)\n\nУчастники: {', '.join(dict.fromkeys(names.values()))}\n\n"
    out.write_text(header + md + "\n", encoding="utf-8")
    if INBOX.exists():
        (INBOX / f"{stamp}_diarized.md").write_text(header + md + "\n", encoding="utf-8")
        mark_webhook_duplicates(stamp)
    print(f"  готово: {out.name}, спикеров {len(set(names.values()))}, реплик {len(lines)}")
    return True


def mark_webhook_duplicates(stamp: str) -> None:
    """Дедуп карточек: тот же разговор приходит и вебхуком приложения (их STT,
    хуже), и нашей диаризацией. Диаризованная версия главнее: вебхук-файлы из
    временного окна записи помечаются разобранными — ночной разбор видит
    разговор один раз. Нет диаризации — вебхук остаётся резервом."""
    from datetime import datetime, timedelta
    try:
        start = datetime.strptime(stamp, "%Y-%m-%d_%H%M%S")
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
    ap.add_argument("--ref", default=str(REF_DEFAULT), help="эталон голоса владельца")
    # мультибазы: у второго носителя своя база — расшифровка должна лечь
    # в его INBOX, а не в дневник владельца
    ap.add_argument("--inbox", default=None, help="куда класть копию расшифровки")
    # своя библиотека голосов на носителя: иначе люди из окружения владельца
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
        sys.exit("HF_TOKEN не задан в .env")
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
