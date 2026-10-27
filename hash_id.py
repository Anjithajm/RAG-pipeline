"""Generate a hash ID from the already-extracted PDF text."""

import hashlib


def get_hash_id(raw_pdf_text: str) -> str:
    return hashlib.sha256(
        raw_pdf_text.encode("utf-8")
    ).hexdigest()