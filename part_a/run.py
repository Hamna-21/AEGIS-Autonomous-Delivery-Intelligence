"""Run the Part A benchmark and environment report.

The implementation remains in the current Part A modules while the project is
being reorganized. This entry point provides the clean folder-based command.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "part_a.py"), run_name="__main__")
