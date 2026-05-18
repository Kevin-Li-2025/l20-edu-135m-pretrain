from __future__ import annotations

import os
import tempfile


def set_default_hf_home() -> None:
    os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "l20-hf-cache"))
