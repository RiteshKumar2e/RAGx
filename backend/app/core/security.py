"""Upload validation and the optional API-key gate.

Validation is defence-in-depth: extension allow-list, declared MIME sanity
check, magic-byte sniffing, size ceiling and filename sanitisation. A file is
only accepted when the extension, the content type and the leading bytes agree.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from pathlib import Path

from fastapi import Header

from app.core.config import get_settings
from app.core.errors import AuthError, FileTooLargeError, UnsupportedFileError

# Extension -> (accepted MIME prefixes, magic byte signatures)
_SIGNATURES: dict[str, tuple[tuple[str, ...], tuple[bytes, ...]]] = {
    ".pdf": (("application/pdf",), (b"%PDF-",)),
    ".docx": (
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        ),
        (b"PK\x03\x04",),
    ),
    ".txt": (("text/",), ()),
    ".md": (("text/",), ()),
    ".csv": (("text/", "application/csv", "application/vnd.ms-excel"), ()),
    ".png": (("image/png",), (b"\x89PNG\r\n\x1a\n",)),
    ".jpg": (("image/jpeg",), (b"\xff\xd8\xff",)),
    ".jpeg": (("image/jpeg",), (b"\xff\xd8\xff",)),
    ".webp": (("image/webp",), (b"RIFF",)),
    ".tiff": (("image/tiff",), (b"II*\x00", b"MM\x00*")),
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_filename(filename: str) -> str:
    """Strip directory components and unsafe characters from an upload name."""
    name = Path(filename or "upload").name
    name = unicodedata.normalize("NFKD", name)
    name = _SAFE_NAME.sub("_", name).strip(" .")
    if not name:
        name = f"upload_{secrets.token_hex(4)}"
    return name[:180]


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_upload(filename: str, content_type: str | None, head: bytes, size: int) -> str:
    """Validate an upload and return the sanitised filename.

    ``head`` should be the first ~512 bytes of the file.
    """
    settings = get_settings()
    safe_name = sanitize_filename(filename)
    ext = file_extension(safe_name)

    if ext not in settings.allowed_extension_set:
        raise UnsupportedFileError(
            f"'{ext or 'unknown'}' is not an accepted file type. "
            f"Allowed: {', '.join(sorted(settings.allowed_extension_set))}"
        )

    if size <= 0:
        raise UnsupportedFileError("The uploaded file is empty.")

    if size > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File is {size / 1_048_576:.1f} MB; the limit is {settings.max_upload_mb} MB."
        )

    mimes, magics = _SIGNATURES.get(ext, ((), ()))

    if content_type and mimes:
        declared = content_type.split(";")[0].strip().lower()
        # ``application/octet-stream`` is what many browsers send for unknown
        # types; we fall through to magic-byte checking in that case.
        if declared != "application/octet-stream" and not any(declared.startswith(m) for m in mimes):
            raise UnsupportedFileError(
                f"Declared content type '{declared}' does not match the '{ext}' extension."
            )

    if magics and not any(head.startswith(sig) for sig in magics):
        raise UnsupportedFileError(
            f"The file contents do not look like a valid '{ext}' file."
        )

    if not magics:
        # Text-family formats: reject files with NUL bytes (binary smuggling).
        if b"\x00" in head:
            raise UnsupportedFileError(f"'{ext}' files must be text, but binary content was found.")

    return safe_name


async def require_api_key(x_ragx_key: str | None = Header(default=None)) -> None:
    """Optional shared-secret gate for mutating endpoints.

    Disabled when ``RAGX_API_KEY`` is unset, which keeps local development
    frictionless while allowing deployments to lock the API down.
    """
    settings = get_settings()
    if not settings.ragx_api_key:
        return
    if not x_ragx_key or not secrets.compare_digest(x_ragx_key, settings.ragx_api_key):
        raise AuthError()


__all__ = ["validate_upload", "sanitize_filename", "file_extension", "require_api_key"]
