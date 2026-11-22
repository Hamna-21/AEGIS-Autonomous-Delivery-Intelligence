"""Single instance-driven configuration for the canonical OEL entry points."""
import os

from dotenv import load_dotenv

load_dotenv()

from common import MEMBERS
from oel_instance import make_instance, stamp

CMS_IDS = [cms for _, cms in MEMBERS]
INSTANCE, GRID = make_instance(CMS_IDS)
INSTANCE_ID = INSTANCE["instance_id"]
THIRD_TWIST = "parcel deadlines: each parcel expires after a configured step budget and incurs a late-delivery penalty"
ACTIVE_TWISTS = list(INSTANCE["twists"]) + [THIRD_TWIST]
GROUP_STAMP = stamp(MEMBERS)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-3-flash-lite",
    "gemini-2.5-flash-lite",
]
GEMINI_CACHE_DIR = "outputs/llm_cache"

__all__ = ["MEMBERS", "CMS_IDS", "INSTANCE", "GRID", "INSTANCE_ID",
           "THIRD_TWIST", "ACTIVE_TWISTS", "GROUP_STAMP", "GEMINI_MODEL",
           "GEMINI_FALLBACK_MODELS", "GEMINI_CACHE_DIR"]
