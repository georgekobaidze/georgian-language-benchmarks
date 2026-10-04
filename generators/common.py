"""Shared helpers for building benchmark data."""

import unicodedata


def normalize(text: str) -> str:
    """Canonical form for comparing answers: Unicode NFC, single spaces, no padding.

    Task files on Kaggle keep their own copy of this function (they can't
    import from the repo), so keep the two in sync.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())
