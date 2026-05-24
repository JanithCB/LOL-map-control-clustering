# Path: tests/test_gui_launch_imports.py
# Summary: Smoke test for GUI module imports.
# Description: Validates that the GUI launcher and main window import successfully without starting the full app.

import pytest

def test_gui_main_window_imports_cleanly():
    """Ensure gui.main_window and MainWindow import successfully."""
    from gui.main_window import MainWindow, main
    assert MainWindow is not None
    assert callable(main)

def test_run_gui_imports_cleanly():
    """Ensure the root-level run_gui.py script imports successfully."""
    import run_gui
    assert hasattr(run_gui, "main")
