"""Single instance-driven configuration for the canonical OEL entry points."""
from common import MEMBERS
from oel_instance import make_instance, stamp

CMS_IDS = [cms for _, cms in MEMBERS]
INSTANCE, GRID = make_instance(CMS_IDS)
INSTANCE_ID = INSTANCE["instance_id"]
THIRD_TWIST = "parcel deadlines: each parcel expires after a configured step budget and incurs a late-delivery penalty"
ACTIVE_TWISTS = list(INSTANCE["twists"]) + [THIRD_TWIST]
GROUP_STAMP = stamp(MEMBERS)
GEMINI_MODEL = "gemini-3.8-flash"

__all__ = ["MEMBERS", "CMS_IDS", "INSTANCE", "GRID", "INSTANCE_ID",
           "THIRD_TWIST", "ACTIVE_TWISTS", "GROUP_STAMP", "GEMINI_MODEL"]
