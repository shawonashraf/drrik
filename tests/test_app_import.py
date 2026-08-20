"""The Gradio app module must import without loading a model or requiring gradio."""

import importlib
import sys
from pathlib import Path

import pytest


def test_app_module_imports_without_model():
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        module = importlib.import_module("eiffel_tower_app")
    finally:
        sys.path.pop(0)

    assert module.DATASET == "shawon/llama-3.1-8b-instruct_eiffel_tower"
    assert module.MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct"
    assert callable(module.build_demo)
    assert callable(module.main)


def test_build_demo_requires_gradio():
    pytest.importorskip("gradio")
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        module = importlib.import_module("eiffel_tower_app")
    finally:
        sys.path.pop(0)

    from unittest.mock import MagicMock

    demo = module.build_demo(MagicMock())
    assert demo is not None
