import os
import sys

def get_path(relative_path: str) -> str:
    """Return absolute path to resource, whether bundled by PyInstaller or not."""
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running from source
        base_path = os.path.abspath(".")
    return os.path.normpath(os.path.join(base_path, relative_path))
