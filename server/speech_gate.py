"""Детектор речи для моста: всплеск над СОБСТВЕННЫМ шумовым полом потока.

Зачем отдельный файл. Старый детектор аудиомоста (`is_silence`) усреднял
амплитуду по ВСЕМУ 8-секундному куску и сравнивал с общей константой 500.
Пока кулон резал тишину сам (VAD на плате), в мост приходила почти сплошная
речь и константа работа. С непрерывным захватом PORT-7 в
мост идёт и фон комнаты, и тогда константа ломается сразу с двух сторон:

- в ТИХОЙ переговорной (шумовой пол ~110) редкая дальняя речь тонет в среднем
  по 8 секундам: кусок с 1 секундой речи даёт среднее ~360 < 500 → «тишина».
  Живой замер на совещании 26.08: при 1 с речи на кусок терялось 65 % кусков,
  при 0,5 с — 97 %. Это дырки в расшифровке И ложное закрытие wav (тихие
  куски копят счётчик 90 с);
- в ОБЫЧНОЙ комнате (пол ~850-1000) и у старой платы второго носителя (пол ~2200) фон САМ
  выше 500, поэтому тишины не бывает НИКОГДА: разговор не закрывается вовсе,
  а в STT круглосуточно едет пустая комната и рождает галлюцинации.

Замеры 26.08 (roles/backend/METHODS.md, «Слух моста»): шумовой пол зависит не
от платы, а от КОМНАТЫ, и гуляет вдесятеро у одного и того же кулона за один
день — 110 в тихой переговорной против 995 в другой записи. Поэтому никакая
константа (даже отдельная на носителя) не годится: порог должен считаться от
самого потока.

Как считает этот детектор:
1. кусок режется на окна по 100 мс, у каждого берётся RMS (громкость);
2. тихая десятая часть окон — оценка ФОНА этого потока прямо сейчас;
3. фон запоминается: вниз идёт быстро (тихий момент — честная улика о фоне),
   вверх — медленно. Оценка берётся по ТИХОЙ десятой части окон, поэтому за
   речью пол не уползает: даже в плотном разговоре есть паузы между словами;
4. речь = хотя бы MIN_WINS окон, которые выше фона в RATIO раз.

ГРАБЛЯ, пойманная на замерах 26.08 (не повторять): «поднимать пол только по
НЕречевому куску» кажется разумным, но ЗАЛИПАЕТ намертво. Дима выходит из
тихой переговорной (пол 141) в шумную комнату — там КАЖДЫЙ кусок выше старого
порога, значит каждый считается речью, значит пол не поднимается никогда.
Замер: пол застревал на 141 навсегда, тишины не находилось ВООБЩЕ (0 % кусков),
то есть wav не закрывался и в STT круглосуточно ехал шум. Поэтому пол растёт
ВСЕГДА. Вторая страховка — `reset()` при закрытии разговора: новый разговор
учит фон заново, с первого куска.

Состояние (фон) — СВОЁ на носителя: у второго носителя своя комната и своя плата.
Все пороги переопределяются через .env, без правки кода.
"""
import math
import os

try:                       # C-скорость; в Python 3.13 модуль удалён
    import audioop
except Exception:          # pragma: no cover — запасной путь на будущее
    audioop = None

def _setting(name: str, legacy: str, default):
    return os.environ.get(name, os.environ.get(legacy, default))


SAMPLE_RATE = int(_setting("AUDIO_BRIDGE_SAMPLE_RATE", "AUDIO_BRIDGE_SAMPLE_RATE", 16000))

# Окно 100 мс: короче — шумит на отдельных слогах, длиннее — теряет короткие реплики.
WIN_MS = int(_setting("AUDIO_BRIDGE_GATE_WIN_MS", "AUDIO_BRIDGE_GATE_WIN_MS", 100))
# Во сколько раз окно должно превысить фон, чтобы считаться речью.
RATIO = float(_setting("AUDIO_BRIDGE_GATE_RATIO", "AUDIO_BRIDGE_GATE_RATIO", 2.2))
# Добавка к порогу — страховка на случай очень тихого фона (тихая переговорная,
# где пол ~110: без добавки порогом стало бы 240 и в речь пролезал бы шорох).
MARGIN = float(_setting("AUDIO_BRIDGE_GATE_MARGIN", "AUDIO_BRIDGE_GATE_MARGIN", 310))
# Ниже этого RMS речи не бывает ни при каком фоне (защита от «цифровой тишины»).
ABS_MIN = float(_setting("AUDIO_BRIDGE_GATE_ABS_MIN", "AUDIO_BRIDGE_GATE_ABS_MIN", 250))
# Сколько окон должно быть «громкими»: 4 окна = 0,4 с речи в 8-секундном куске.
MIN_WINS = int(_setting("AUDIO_BRIDGE_GATE_MIN_WINS", "AUDIO_BRIDGE_GATE_MIN_WINS", 4))
# Скорость слежения за фоном: вниз быстро, вверх медленно.
DOWN = float(_setting("AUDIO_BRIDGE_GATE_DOWN", "AUDIO_BRIDGE_GATE_DOWN", 0.35))
UP = float(_setting("AUDIO_BRIDGE_GATE_UP", "AUDIO_BRIDGE_GATE_UP", 0.08))
# Границы фона — чтобы оценка не убежала ни в ноль, ни в бесконечность.
FLOOR_MIN = float(_setting("AUDIO_BRIDGE_GATE_FLOOR_MIN", "AUDIO_BRIDGE_GATE_FLOOR_MIN", 50))
FLOOR_MAX = float(_setting("AUDIO_BRIDGE_GATE_FLOOR_MAX", "AUDIO_BRIDGE_GATE_FLOOR_MAX", 5000))


def window_rms(pcm: bytes, win_ms: int = WIN_MS) -> list[float]:
    """Громкость (RMS) по окнам win_ms. PCM16 моно."""
    step = SAMPLE_RATE * win_ms // 1000 * 2      # байт в окне
    out = []
    for i in range(0, len(pcm) - step + 1, step):
        w = pcm[i:i + step]
        if audioop is not None:
            out.append(float(audioop.rms(w, 2)))
        else:                                     # pragma: no cover
            # запасной путь без audioop: RMS по каждому 4-му сэмплу — та же
            # величина с погрешностью ~3 %, порогов не сдвигает
            tot = n = 0
            for j in range(0, len(w) - 1, 8):
                s = int.from_bytes(w[j:j + 2], "little", signed=True)
                tot += s * s
                n += 1
            out.append(math.sqrt(tot / n) if n else 0.0)
    if not out and pcm:                           # кусок короче одного окна
        if audioop is not None:
            out.append(float(audioop.rms(pcm[:len(pcm) // 2 * 2], 2)))
        else:                                     # pragma: no cover
            out.append(0.0)
    return out


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    return s[min(len(s) - 1, int(len(s) * pct / 100))]


class SpeechGate:
    """Детектор речи ОДНОГО потока. Экземпляр — на носителя (у каждого свой фон)."""

    def __init__(self):
        self.floor: float | None = None

    def reset(self):
        """Разговор закрылся — следующий может быть в другой комнате.
        Забываем фон, чтобы он выучился заново с первого куска."""
        self.floor = None

    def is_silence(self, pcm: bytes) -> bool:
        """True = в куске речи нет (в STT не отправлять, копить тишину разговора)."""
        if not pcm:
            return True
        wins = window_rms(pcm)
        if not wins:
            return True
        cand = _percentile(wins, 10)          # тихая десятая часть = оценка фона
        if self.floor is None:                # первый кусок сессии задаёт фон
            self.floor = min(max(cand, FLOOR_MIN), FLOOR_MAX)
        thr = max(self.floor * RATIO, self.floor + MARGIN, ABS_MIN)
        # на коротком хвосте буфера окон мало — там хватит и одного громкого
        need = max(1, min(MIN_WINS, len(wins) // 3))
        speech = sum(1 for w in wins if w > thr) >= need
        # фон: вниз быстро, вверх медленно — но ВСЕГДА (см. граблю в шапке)
        if cand < self.floor:
            self.floor += (cand - self.floor) * DOWN
        else:
            self.floor += (cand - self.floor) * UP
        self.floor = min(max(self.floor, FLOOR_MIN), FLOOR_MAX)
        return not speech
