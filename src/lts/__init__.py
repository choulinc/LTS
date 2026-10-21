"""Lightweight Transolver Surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models.factory import LTSConfig, build_lts

__all__ = ["LTSConfig", "build_lts"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load GPU-heavy model dependencies only when the model API is used."""

    if name in __all__:
        from .models.factory import LTSConfig, build_lts

        return {"LTSConfig": LTSConfig, "build_lts": build_lts}[name]
    raise AttributeError(name)
