"""Validated Part D tools with a real Gemini LangChain command adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from config import GEMINI_CACHE_DIR, GEMINI_FALLBACK_MODELS, GROUP_STAMP, INSTANCE_ID
from part_b.search import DeliveryProblem, a_star, bfs, greedy_best_first_search, uniform_cost_search
from world.fog import observe

ALGORITHMS = {"bfs": bfs, "ucs": uniform_cost_search, "uniform cost": uniform_cost_search,
              "a*": a_star, "astar": a_star, "greedy": greedy_best_first_search}
TOOLS = {"plan_route", "simulate", "observe_fog"}


@dataclass
class Event:
    timestamp: str
    phase: str
    name: str
    arguments: dict
    observation: dict


class ValidationError(ValueError):
    pass


def default_grid():
    grid = np.zeros((7, 7), dtype=int)
    for cell in ((1, 2), (2, 2), (4, 4), (5, 4)):
        grid[cell] = -1
    return grid


class AEGISCommandInterface:
    """Natural-language interface backed by Gemini through LangChain.

    Direct tool methods remain deterministic and do not require a key. The
    natural-language ``handle`` method refuses to run without Gemini, so no
    offline parser result can be mistaken for an LLM result.
    """

    def __init__(self, grid=None, terrain_cost=None, log_path=None, cache_dir=None):
        self.grid = np.asarray(grid if grid is not None else default_grid(), dtype=int)
        self.terrain_cost = terrain_cost or {"road": 1, "grass": 3, "mud": 7, "ford": 8}
        self.events = []
        self.log_path = Path(log_path) if log_path else None
        self.api_key_available = bool(os.getenv("GOOGLE_API_KEY"))
        self.mode = "gemini_langchain" if self.api_key_available else "unavailable"
        self.models = list(dict.fromkeys(GEMINI_FALLBACK_MODELS))
        self.cache_dir = Path(cache_dir or GEMINI_CACHE_DIR)
        self.llms = {}
        self.last_model = ""
        self.last_cache_hit = False
        if self.api_key_available:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._chat_model_type = ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ValidationError("Install langchain-google-genai to use Gemini Part D") from exc

    def _cache_path(self, model, request, validation):
        material = json.dumps({"model": model, "request": request,
                               "validation": bool(validation)}, sort_keys=True).encode()
        return self.cache_dir / f"{hashlib.sha256(material).hexdigest()}.json"

    def _llm_for(self, model):
        if model not in self.llms:
            self.llms[model] = self._chat_model_type(
                model=model, temperature=0, google_api_key=os.environ["GOOGLE_API_KEY"],
                timeout=30, max_retries=0
            )
        return self.llms[model]

    def _save_cache(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _rate_limited(self, error):
        text = str(error).lower()
        return "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text

    def _event(self, phase, name, arguments, observation):
        observation = dict(observation or {})
        observation.setdefault("stamp", GROUP_STAMP)
        observation.setdefault("instance_id", INSTANCE_ID)
        event = Event(datetime.now(timezone.utc).isoformat(), phase, name,
                      self._jsonable(arguments), self._jsonable(observation))
        self.events.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(event)) + "\n")

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _cell(self, value, label):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(not isinstance(x, int) for x in value):
            raise ValidationError(f"{label} must be [row, col]")
        cell = tuple(value)
        if not (0 <= cell[0] < self.grid.shape[0] and 0 <= cell[1] < self.grid.shape[1]):
            raise ValidationError(f"{label} is outside the grid")
        if self.grid[cell] < 0:
            raise ValidationError(f"{label} is an obstacle")
        return cell

    def _problem(self, start, stops, flooded_cells=(), flood_enabled=False, flood_interval=5):
        start = self._cell(start, "start")
        if not isinstance(stops, list) or not stops:
            raise ValidationError("stops must be a non-empty list")
        stops = [self._cell(stop, "stop") for stop in stops]
        flooded = [self._cell(cell, "flooded cell") for cell in flooded_cells]
        if not isinstance(flood_interval, int) or flood_interval <= 0:
            raise ValidationError("flood_interval must be a positive integer")
        return DeliveryProblem(self.grid, start, stops, self.terrain_cost,
                               flooded_cells=flooded, flood_enabled=bool(flood_enabled),
                               flood_interval=flood_interval)

    def plan_route(self, start, stops, algorithm="a*", flooded_cells=None,
                   flood_enabled=False, flood_interval=5):
        args = {"start": start, "stops": stops, "algorithm": algorithm,
                "flooded_cells": flooded_cells or [], "flood_enabled": flood_enabled,
                "flood_interval": flood_interval}
        self._event("Sense", "plan_route.request", args, {"instance_id": INSTANCE_ID})
        key = str(algorithm).lower().strip()
        if key not in ALGORITHMS:
            obs = {"ok": False, "error": f"unsupported algorithm: {algorithm}"}
            self._event("Reason", "plan_route.validation", args, obs)
            raise ValidationError(obs["error"])
        problem = self._problem(start, stops, flooded_cells or [], flood_enabled, flood_interval)
        result = ALGORITHMS[key](problem)
        observation = {"found": result.found, "cost": result.cost if result.found else None,
                       "path": result.path if result.found else [], "nodes_expanded": result.nodes_expanded,
                       "runtime_seconds": result.runtime_seconds, "instance_id": INSTANCE_ID}
        self._event("Reason", "plan_route.search", args, observation)
        if not result.found:
            raise ValidationError("no route exists for the requested delivery")
        self._event("Act", "plan_route.result", args, observation)
        return {"ok": True, "mode": self.mode, "algorithm": result.algorithm, **observation}

    def observe_fog(self, position, radius=4):
        position = self._cell(position, "position")
        if not isinstance(radius, int) or radius < 0:
            raise ValidationError("radius must be a non-negative integer")
        args = {"position": position, "radius": radius}
        self._event("Sense", "observe_fog.request", args, {"instance_id": INSTANCE_ID})
        observed, mask = observe(self.grid, position, radius)
        observation = {"ok": True, "position": position, "radius": radius,
                       "visible_cells": int(mask.sum()), "grid": observed.tolist(),
                       "instance_id": INSTANCE_ID}
        self._event("Act", "observe_fog.result", args, observation)
        return observation

    def simulate(self, start, stops, slip_probability=0.0, seed=0, algorithm="a*",
                 flooded_cells=None, flood_enabled=False, flood_interval=5):
        if not isinstance(slip_probability, (int, float)) or not 0 <= slip_probability <= 1:
            raise ValidationError("slip_probability must be between 0 and 1")
        args = {"start": start, "stops": stops, "slip_probability": slip_probability,
                "seed": seed, "algorithm": algorithm}
        self._event("Sense", "simulate.request", args, {"instance_id": INSTANCE_ID})
        plan = self.plan_route(start, stops, algorithm, flooded_cells, flood_enabled, flood_interval)
        rng = random.Random(seed)
        actual = list(plan["path"])
        slips = 0
        if slip_probability:
            for index in range(1, len(actual)):
                if rng.random() < slip_probability:
                    slips += 1
                    actual[index] = actual[index - 1]
        observation = {"ok": True, "planned_cost": plan["cost"], "planned_path": plan["path"],
                       "executed_path": actual, "slips": slips, "replanned": bool(slips),
                       "final_position": actual[-1], "instance_id": INSTANCE_ID}
        self._event("Reason", "simulate.execution", args, observation)
        self._event("Act", "simulate.result", args, observation)
        return observation

    def _parse_llm_content(self, content):
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        content = str(content).strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValidationError("Gemini returned invalid JSON") from exc
        if not isinstance(parsed, dict) or parsed.get("error"):
            raise ValidationError(str(parsed.get("error", "Gemini returned an invalid command")))
        if parsed.get("tool") not in TOOLS or not isinstance(parsed.get("args"), dict):
            raise ValidationError("Gemini returned an unsupported or malformed tool call")
        return parsed

    def _llm_request(self, request, validation=True):
        if not self.api_key_available:
            raise ValidationError("GOOGLE_API_KEY is not set; Part D was skipped. Set it before using natural-language commands.")
        from langchain_core.messages import HumanMessage, SystemMessage
        validation_text = "Validate that all requested values are explicit and inside the supplied grid." if validation else "Extract only explicit values. The execution layer will still enforce safety validation."
        system = f"""You are the AEGIS command parser. Return JSON only, with no markdown.
Use exactly one tool: plan_route, simulate, or observe_fog.
JSON shape: {{\"tool\": string, \"args\": object}}.
For plan_route and simulate, args must include start as [row,col], stops as a non-empty list of [row,col], and algorithm from {sorted(ALGORITHMS)}.
For simulate, optionally include slip_probability between 0 and 1, seed, flooded_cells, flood_enabled, flood_interval.
For observe_fog, args must include position as [row,col] and may include radius.
The grid shape is {list(self.grid.shape)}. {validation_text}
Never invent a result. If the request is missing required values, return {{\"error\": \"missing required values\"}}.
Instance ID is {INSTANCE_ID}."""
        self.last_model = ""
        self.last_cache_hit = False
        for model in self.models:
            cache_path = self._cache_path(model, request, validation)
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.last_model = model
                self.last_cache_hit = True
                if cached.get("status") == "rate_limited":
                    cache_path.unlink(missing_ok=True)
                    self.last_cache_hit = False
                    continue
                if cached.get("status") != "success":
                    raise ValidationError(cached.get("error", "Cached Gemini request failed"))
                return self._parse_llm_content(cached["content"])
            try:
                response = self._llm_for(model).invoke([SystemMessage(content=system), HumanMessage(content=request)])
                content = response.content
                self._save_cache(cache_path, {"status": "success", "model": model,
                                               "request": request, "validation": bool(validation),
                                               "content": content})
                self.last_model = model
                return self._parse_llm_content(content)
            except Exception as exc:
                limited = self._rate_limited(exc)
                if limited:
                    print(f"Gemini model exhausted: {model}. Trying the next configured model.")
                    continue
                self._save_cache(cache_path, {"status": "error", "model": model,
                                               "request": request, "validation": bool(validation),
                                               "error": str(exc)})
                raise ValidationError(f"Gemini request failed for model {model}: {exc}") from exc
        raise ValidationError("All configured Gemini models are exhausted. Part D stopped without fake results.")

    def handle(self, request, validation=True):
        if not isinstance(request, str) or not request.strip():
            raise ValidationError("request must be non-empty text")
        self._event("Sense", "user.request", {"text": request}, {"mode": self.mode, "instance_id": INSTANCE_ID})
        parsed = self._llm_request(request, validation=validation)
        self._event("Reason", "gemini_langchain.tool_call", {"text": request, "validation": validation,
                                                               "model": self.last_model, "cache_hit": self.last_cache_hit}, parsed)
        result = getattr(self, parsed["tool"])(**parsed["args"])
        self._event("Act", "gemini_langchain.result", parsed["args"], result)
        return {"mode": self.mode, "request": request, "parsed": parsed, "result": result,
                "model": self.last_model, "cache_hit": self.last_cache_hit, "validation": validation,
                "trace": [asdict(event) for event in self.events[-8:]]}

    def metrics(self, expected):
        if not expected:
            return {"requests": 0, "success_rate": 0.0, "mean_cost": None, "mean_tool_calls": 0.0}
        successes = sum(1 for item in expected if item["success"])
        costs = [item["cost"] for item in expected if item["cost"] is not None]
        return {"requests": len(expected), "success_rate": successes / len(expected),
                "mean_cost": sum(costs) / len(costs) if costs else None,
                "mean_tool_calls": sum(item["tool_calls"] for item in expected) / len(expected)}
