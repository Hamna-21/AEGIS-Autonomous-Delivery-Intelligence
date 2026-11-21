# Part D natural-language command interface

This implementation uses an equivalent deterministic command-router approach
because this environment has no configured API key and no LangChain dependency.
It is explicitly labelled `offline_parser`, not an LLM. The parser understands
coordinate delivery requests such as:

`Plan a route from (0,0) to (6,6) and (0,1) using A*`

The interface exposes three validated tools: `plan_route`, `simulate`, and
`spread_flood`. `plan_route` calls the actual Part B search implementations.
`simulate` executes the planned path with optional slip. `spread_flood` wraps
the Part A flood transition as the custom environmental tool.

Every request, parsed command, tool call, arguments, and observation is logged
with Sense, Reason, or Act phases. Failed validation and impossible searches
raise `ValidationError` and do not fabricate a successful result.

The evaluation script compares parsed requests with direct planner calls,
reports success rate, average cost, and average tool calls, and checks 20
natural-language requests. Set `OPENAI_API_KEY` only when adding a real
provider adapter. The current mode remains explicitly offline.
