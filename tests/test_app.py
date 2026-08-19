from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_without_exceptions() -> None:
    script = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(script).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == (
        "A Label-Free Deep Learning Framework for Image Fusion Through Self-Supervised "
        "Feature Decomposition"
    )
    assert app.info[0].value == "Upload two aligned source images to begin."
