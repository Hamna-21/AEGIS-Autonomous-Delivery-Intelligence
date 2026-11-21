"""Evaluate the Part D offline parser against direct Part B planning."""
from part_d import AEGISCommandInterface, ValidationError
from part_d.interface import ALGORITHMS


REQUESTS = [
    "Plan a route from (0,0) to (6,6) using A*",
    "Deliver from (0,0) to (0,1) and (6,6) with BFS",
    "Find a route from (6,0) to (0,0) using uniform cost",
    "Use greedy to deliver from (6,0) to (0,1)",
    "Simulate delivery from (0,0) to (6,6) with slip",
    "Execute from (6,0) to (0,0) with slip",
    "Plan from (1,1) to (3,3) with A*",
    "Deliver from (2,0) to (6,6) using BFS",
    "Route from (3,3) to (0,0) using A*",
    "Simulate from (0,6) to (6,0)",
    "Plan a route from (6,6) to (0,1) using A*",
    "Deliver from (0,1) to (6,6) with greedy",
    "Find a path from (5,5) to (0,0) using ucs",
    "Execute delivery from (0,0) to (5,6) with slip",
    "Plan from (4,0) to (6,6) using A*",
    "Route from (6,1) to (0,6) with BFS",
    "Simulate delivery from (1,1) to (6,6) with slip",
    "Plan a route from (3,0) to (0,6) using A*",
    "Deliver from (6,5) to (0,0) with A*",
    "Find a route from (2,5) to (6,0) using greedy",
]


def main():
    interface = AEGISCommandInterface(log_path="outputs/part_d_events.jsonl")
    rows = []
    direct_matches = 0
    for request in REQUESTS:
        before = len(interface.events)
        try:
            response = interface.handle(request)
            result = response["result"]
            parsed_args = response["parsed"]["args"]
            direct = ALGORITHMS[parsed_args["algorithm"]](
                interface._problem(parsed_args["start"], parsed_args["stops"]))
            reported_cost = result.get("cost", result.get("planned_cost"))
            direct_matches += int(reported_cost == direct.cost)
            rows.append({"request": request, "success": result["ok"],
                         "cost": result.get("cost", result.get("planned_cost")),
                         "tool_calls": len(interface.events) - before})
        except ValidationError as error:
            rows.append({"request": request, "success": False, "cost": None,
                         "tool_calls": len(interface.events) - before, "error": str(error)})
    print("mode", interface.mode)
    print(interface.metrics(rows))
    print("requests_tested", len(rows))
    print("direct_planner_matches", f"{direct_matches}/{len(rows)}")
    print("direct_planner_comparison: offline parsed requests were checked against direct same-algorithm calls")


if __name__ == "__main__":
    main()
