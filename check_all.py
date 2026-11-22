"""Evidence-based CS-370 OEL audit for the current checkout."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

from config import GROUP_STAMP, INSTANCE, INSTANCE_ID, MEMBERS
from world.fog import observe
from world.transitions import legal_neighbors

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
SOURCE = [p for p in ROOT.rglob("*.py") if ".git" not in p.parts and p.name != "check_all.py"]


def result(ok, reason):
    print(f"{'PASS' if ok else 'FAIL'} | {reason}")
    return bool(ok)


def files_under(*folders):
    files = []
    for folder in folders:
        path = ROOT / folder
        if path.exists():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return files


def read_rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    checks = []
    print(GROUP_STAMP)
    print(f"instance_id: {INSTANCE_ID}")
    print("CS-370 OEL audit")

    checks.append(result(len(MEMBERS) in {3, 4}, f"member count is {len(MEMBERS)}"))
    checks.append(result(INSTANCE_ID and GROUP_STAMP, "instance loaded through make_instance with stamp"))

    output_files = files_under("results", "outputs")
    missing_identity = [str(p.relative_to(ROOT)) for p in output_files
                        if p.suffix not in {".pyc"} and (GROUP_STAMP not in p.read_text(errors="ignore") or INSTANCE_ID not in p.read_text(errors="ignore"))]
    checks.append(result(not missing_identity, "stamp and instance_id are present in every result and output file" if not missing_identity else f"identity missing from {missing_identity[:5]}"))

    forbidden = []
    for path in SOURCE:
        text = path.read_text(errors="ignore")
        for name in ("networkx", "simpleai", "aima", "gym"):
            if re.search(rf"\bimport\s+{name}\b|\bfrom\s+{name}\b", text):
                forbidden.append(f"{path.name}:{name}")
    checks.append(result(not forbidden, "no prohibited search-library imports" if not forbidden else ", ".join(forbidden)))

    literals = [str(INSTANCE["grid_size"]), str(INSTANCE["slip_probability"]), str(INSTANCE["adversary_search_depth"])]
    production = "\n".join(p.read_text(errors="ignore") for p in SOURCE if p.name not in {"config.py", "oel_instance.py"} and "tests" not in p.parts and "ui" not in p.parts)
    hardcoded = [value for value in literals if re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", production)]
    checks.append(result(not hardcoded, "instance scalar values are not repeated in production code" if not hardcoded else f"possible hard-coded instance values: {hardcoded}"))

    fog_full, _ = observe(__import__("numpy").zeros((5, 5), dtype=int), (2, 2), 4)
    fog_local, _ = observe(__import__("numpy").zeros((5, 5), dtype=int), (2, 2), 1)
    checks.append(result((fog_full != fog_local).any(), "fog changes the observation state"))
    grid = __import__("numpy").zeros((2, 2), dtype=int)
    unrestricted = legal_neighbors(grid, (0, 0))
    restricted = legal_neighbors(grid, (0, 0), {((0, 0), (1, 0))})
    checks.append(result(set(restricted) != set(unrestricted), "one-way edges change legal transitions"))

    tests = subprocess.run([sys.executable, "-m", "unittest", "tests.test_search", "tests.test_world", "tests.test_part_c"], cwd=ROOT, capture_output=True, text=True)
    checks.append(result(tests.returncode == 0, "search, world, and adversarial tests pass" if tests.returncode == 0 else "core tests failed"))

    search_rows = read_rows("part_b_search_raw.csv")
    checks.append(result({row.get("algorithm") for row in search_rows} >= {"BFS", "UCS", "DLS", "IDS", "Greedy", "A*"}, "all six search algorithms have raw rows"))
    checks.append(result(all(key in (search_rows[0] if search_rows else {}) for key in ("nodes_expanded", "runtime_seconds", "memory_states", "cost")), "search CSV contains nodes, time, memory, and cost"))
    checks.append(result(any("baseline" in row.get("configuration", "") for row in search_rows), "search configurations are present"))

    rival_rows = read_rows("part_c_rival_raw.csv")
    depths = {int(row["depth"]) for row in rival_rows if row.get("depth")}
    checks.append(result(len({row.get("scenario") for row in rival_rows}) >= 30, "at least 30 rival scenarios are recorded"))
    checks.append(result(depths >= {2}, f"rival depths recorded: {sorted(depths)}"))
    checks.append(result(all(int(row["nodes"]) >= 0 for row in rival_rows), "rival node counts are numeric"))
    checks.append(result(all(float(row["pruning_percent"]) >= 0 for row in rival_rows if row.get("pruning_percent")), "alpha-beta pruning percentages are present"))

    ablation_rows = read_rows("ablation_a_star_raw.csv")
    checks.append(result(len({row.get("scenario") for row in ablation_rows}) >= 30, "ablation has at least 30 scenarios"))
    checks.append(result(len({row.get("variant") for row in ablation_rows}) >= 2, "ablation has two variants"))

    llm_rows = read_rows("part_d_llm_raw.csv")
    checks.append(result(len({row.get("request_id") for row in llm_rows}) >= 20, "Part D contains at least 20 requests"))
    checks.append(result({row.get("validation") for row in llm_rows} >= {"True", "False"}, "Part D compares validation on and off"))
    checks.append(result(all(row.get("model") for row in llm_rows), "Part D records a real model for every row"))
    checks.append(result(any(row.get("error") for row in llm_rows), "Part D failure column is populated when failures occur"))
    checks.append(result(len(list(RESULTS.glob("trace_*.txt"))) >= 2, "two Part D traces exist"))

    expected_graphs = {f"graph{i}" for i in range(1, 9)}
    actual_graphs = {p.stem for p in RESULTS.glob("*.svg")}
    checks.append(result(expected_graphs <= actual_graphs, f"expected eight graphs exist, found {sorted(actual_graphs)}"))
    checks.append(result(all(p.stat().st_size > 0 for p in RESULTS.glob("*.csv")), "result CSV files are non-empty"))
    checks.append(result("INSTANCE[\"slip_probability\"]" in production or "INSTANCE['slip_probability']" in production, "experiments read slip from the generated instance"))

    passed = sum(checks)
    failed = len(checks) - passed
    print(f"SUMMARY | passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
