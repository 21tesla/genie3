"""Validation helpers for unified Genie3 experiment configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigError(ValueError):
    """Raised when a unified experiment config is invalid."""


def require_mapping(data: Any, name: str) -> dict[str, Any]:
    """Return a shallow dict copy if `data` is a mapping."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"Expected `{name}` to be a mapping.")
    return dict(data)


def optional_mapping(data: Any, name: str) -> dict[str, Any]:
    """Return `{}` for omitted sections and validate present mappings."""
    if data is None:
        return {}
    return require_mapping(data, name)


def require_nonempty_string(value: Any, name: str) -> str:
    """Validate that a required field is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected `{name}` to be a non-empty string.")
    return value


def require_keys(data: Mapping[str, Any], section: str, keys: tuple[str, ...]) -> None:
    """Validate that `section` contains all required keys."""
    missing = [key for key in keys if key not in data or data[key] is None]
    if missing:
        joined = ", ".join(f"`{section}.{key}`" for key in missing)
        raise ConfigError(f"Missing required config field(s): {joined}.")
