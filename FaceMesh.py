import runpy
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).with_name("Face Mesh.py")
    runpy.run_path(str(script_path), run_name="__main__")
