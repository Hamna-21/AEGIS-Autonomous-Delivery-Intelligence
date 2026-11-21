from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INSTANCE_PATH = DATA_DIR / "instance.json"

MEMBERS = [
    ("Khansa Farooq", "471827"),
    ("Sara Amin", "462417"),
    ("Hamna Mushtaq", "486820"),
    ("Faiza Malik", "475313"),
]


def stamp():
    return " | ".join(f"{name} ({cms})" for name, cms in MEMBERS)


def load_instance(path=INSTANCE_PATH):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)
