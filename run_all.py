"""Run the complete OEL pipeline and publish outputs under results/."""
from pathlib import Path
import os
import shutil

from experiments.run_all import main as run_research
from config import GROUP_STAMP, INSTANCE_ID

ROOT = Path(__file__).resolve().parent


def main():
    run_research()
    source = ROOT / "outputs" / "research"
    target = ROOT / "results"
    target.mkdir(exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_file():
            shutil.copy2(item, destination)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Part D traces skipped: GOOGLE_API_KEY is not set.")
    if not os.getenv("GOOGLE_API_KEY"):
        for name in ("part_d_llm_raw.csv", "part_d_events.jsonl", "trace_1.txt", "trace_2.txt"):
            (target / name).unlink(missing_ok=True)
    print(f"stamp: {GROUP_STAMP}")
    print(f"instance_id: {INSTANCE_ID}")
    print(f"results: {target}")


if __name__ == "__main__":
    main()
