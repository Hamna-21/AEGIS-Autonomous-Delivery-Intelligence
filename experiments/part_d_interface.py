"""Run the 20 generated Gemini requests against direct Part B planning."""
from config import GRID, GROUP_STAMP, INSTANCE_ID
from experiments.run_all import generated_requests
from part_d import AEGISCommandInterface, ValidationError
from part_d.interface import ALGORITHMS


def main():
    interface = AEGISCommandInterface(grid=GRID, log_path="outputs/part_d_events.jsonl")
    rows = []
    direct_matches = 0
    for request in generated_requests(GRID):
        before = len(interface.events)
        try:
            response = interface.handle(request)
            result = response["result"]
            args = response["parsed"]["args"]
            direct = ALGORITHMS[args["algorithm"]](interface._problem(args["start"], args["stops"]))
            reported_cost = result.get("cost", result.get("planned_cost"))
            direct_matches += int(reported_cost == direct.cost)
            rows.append({"request": request, "success": result["ok"], "cost": reported_cost,
                         "model": response.get("model"), "tool_calls": len(interface.events) - before})
        except ValidationError as error:
            rows.append({"request": request, "success": False, "cost": None,
                         "model": interface.last_model, "tool_calls": len(interface.events) - before,
                         "error": str(error)})
    print(GROUP_STAMP)
    print(f"instance_id: {INSTANCE_ID}")
    print("mode", interface.mode)
    print(interface.metrics(rows))
    print("requests_tested", len(rows))
    print("direct_planner_matches", f"{direct_matches}/{len(rows)}")


if __name__ == "__main__":
    main()
