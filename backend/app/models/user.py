"""User and system-configuration tables.

Authentication is optional in RAGX (the API-key gate in ``app.core.security``
covers deployment lock-down). The user table exists so documents, conversations
and queries can be attributed and so multi-user deployments have a place to
grow into; a single ``local`` user is seeded at startup.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin

LOCAL_USER_ID = "00000000000000000000000000000001"


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(160), default="Local User")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class SystemSetting(Base, TimestampMixin):
    """Operator-tunable runtime settings.

    Only non-sensitive knobs live here (retrieval parameters, provider *choice*,
    feature toggles). Credentials always come from the environment.
    """

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    description: Mapped[str | None] = mapped_column(Text)


__all__ = ["User", "SystemSetting", "LOCAL_USER_ID"]
