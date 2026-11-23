"""numpy, imported on first use: ``from ._numpy import np`` in place of ``import numpy as np``.

WHY (io-diagnostics, 2026-10-28): numpy is the ``stock`` extra, not a runtime dependency. Without it every module of
the package still imports, every stock Spec still registers (so a page can list and configure them), and only running a
diagnostic that does arithmetic raises ``ModuleNotFoundError`` (name ``numpy``), which the host reports as not_run.
"""

from __future__ import annotations

MISSING = ("numpy is not installed, so the stock diagnostics cannot compute: install the stock extra "
           "(pip install 'robot-diagnostics[stock]')")


class _Numpy:
    def __getattr__(self, name: str):
        import numpy  # raises ModuleNotFoundError(name="numpy") when the extra is absent

        value = getattr(numpy, name)
        setattr(self, name, value)  # the next access is a plain attribute
        return value


def available() -> bool:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


np = _Numpy()

__all__ = ["np", "available", "MISSING"]
