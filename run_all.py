"""Run the complete OEL pipeline and publish outputs under results/."""
from pathlib import Path
import os
import shutil

from experiments.run_all import main as run_research
from experiments.run_all import generated_requests
from part_d.interface import AEGISCommandInterface, default_grid
from config import GRID, GROUP_STAMP, INSTANCE_ID

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
    if os.getenv("GOOGLE_API_KEY"):
        benchmark_grid = default_grid()
        agent = AEGISCommandInterface(grid=benchmark_grid)
        for index, request in enumerate(generated_requests(benchmark_grid)[:2], 1):
            response = agent.handle(request)
            trace = target / f"trace_{index}.txt"
            lines = [f"stamp: {GROUP_STAMP}", f"instance_id: {INSTANCE_ID}", f"request: {request}"]
            for event in response.get("trace", []):
                lines.append(f"{event['phase']} | {event['name']} | {event['observation']}")
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print("Part D traces skipped: GOOGLE_API_KEY is not set.")
        for name in ("part_d_llm_raw.csv", "part_d_events.jsonl", "trace_1.txt", "trace_2.txt"):
            (target / name).unlink(missing_ok=True)
    print(f"stamp: {GROUP_STAMP}")
    print(f"instance_id: {INSTANCE_ID}")
    print(f"results: {target}")


if __name__ == "__main__":
    main()
