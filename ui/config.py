"""Frontend configuration and shared application context."""
from dataclasses import dataclass
from pathlib import Path

from common import load_instance

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = load_instance()
PAGES = ["Mission Control", "AI Assistant", "Algorithm Laboratory", "Rival Intelligence",
         "Environment Laboratory", "Research Analytics", "Reports", "System Architecture", "Settings"]


@dataclass(frozen=True)
class AppContext:
    grid: object
    start: tuple
    rival: tuple
    stops: list
