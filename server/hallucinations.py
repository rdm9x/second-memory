"""Общий фильтр галлюцинаций STT на тишине/шуме.
Используют аудиомост и diarize.py."""

HALLUCINATION_MARKERS = (
    "субтитр", "dimatorzok", "dima torzok", "продолжение следует",
    "спасибо за просмотр", "amara.org", "редактор субтитров",
    "корректор", "подписывайтесь", "ставьте лайк",
)


def is_hallucination(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in HALLUCINATION_MARKERS)
