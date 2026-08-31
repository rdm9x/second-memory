"""Общий фильтр галлюцинаций whisper (титры из обучающих данных на тишине/шуме).
Используют audio_bridge_core.py (live-мост) и diarize.py.
«DimaTorzok» — известный артефакт русской модели whisper (подпись автора
субтитров из обучающих данных), не человек из жизни владельца."""

HALLUCINATION_MARKERS = (
    "субтитр", "dimatorzok", "dima torzok", "продолжение следует",
    "спасибо за просмотр", "amara.org", "редактор субтитров",
    "корректор", "подписывайтесь", "ставьте лайк",
)


def is_hallucination(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in HALLUCINATION_MARKERS)
