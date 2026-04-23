"""
PV Recommendation JSON Schema and validator.

Defines the structured-output schema that Grok (or any backend) must
conform to, plus helpers for validation and repair prompting.

The top-level object has four main keys:
  - "optimal"                  : technically best PV system (ignores budget if needed)
  - "recommended"              : balanced, budget-aware PV system the homeowner should buy
  - "battery_recommendation"   : whether to add a battery and why
  - "panel_brand_recommendation": which brand was selected and why (always populated)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# ── Single-scenario sub-schema ───────────────────────────────
_SCENARIO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "panels": {"type": "integer"},
        "kw_dc": {"type": "number"},
        "target_offset_fraction": {"type": "number"},
        "expected_annual_production_kwh": {"type": "number"},
        "annual_consumption_kwh_used": {"type": "number"},
        "expected_annual_savings_usd": {"type": "number"},
        "capex_estimate_usd": {"type": "number"},
        "payback_years_estimate": {"type": "number"},
        "rationale": {"type": "string"},
        "constraints": {
            "type": "object",
            "properties": {
                "budget_usd": {"type": "number"},
                "max_panels_within_budget": {"type": "integer"},
                "budget_binding": {"type": "boolean"},
            },
            "required": ["budget_usd", "max_panels_within_budget", "budget_binding"],
        },
        "assumptions": {
            "type": "object",
            "properties": {
                "panel_watt_peak": {"type": "number"},
                "system_derate": {"type": "number"},
                "price_per_kwh": {"type": "number"},
            },
            "required": ["panel_watt_peak", "system_derate", "price_per_kwh"],
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "panels",
        "kw_dc",
        "target_offset_fraction",
        "expected_annual_production_kwh",
        "annual_consumption_kwh_used",
        "expected_annual_savings_usd",
        "capex_estimate_usd",
        "payback_years_estimate",
        "rationale",
        "constraints",
        "assumptions",
        "risks",
        "confidence",
    ],
}

# ── Battery recommendation sub-schema ────────────────────────
_BATTERY_RECOMMENDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["add_battery", "evaluate_later", "pv_only"],
        },
        "battery_manufacturer":          {"type": "string"},
        "battery_model":                 {"type": "string"},
        "battery_capacity_kwh":          {"type": "number"},
        "battery_gross_cost_usd":        {"type": "number"},
        "net_battery_cost_after_itc_usd":{"type": "number"},
        "extra_annual_savings_usd":      {"type": "number"},
        "import_reduction_kwh":          {"type": "number"},
        "self_consumption_pct":          {"type": "number"},
        "battery_incremental_payback_years": {"type": ["number", "null"]},
        "rationale":                     {"type": "string"},
    },
    "required": [
        "decision",
        "battery_manufacturer",
        "battery_model",
        "battery_capacity_kwh",
        "battery_gross_cost_usd",
        "net_battery_cost_after_itc_usd",
        "extra_annual_savings_usd",
        "import_reduction_kwh",
        "self_consumption_pct",
        "battery_incremental_payback_years",
        "rationale",
    ],
}

# ── Panel brand recommendation sub-schema ─────────────────────
_PANEL_BRAND_RECOMMENDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selection_mode": {
            "type": "string",
            "enum": ["auto", "user_specified"],
        },
        "selected_manufacturer": {"type": "string"},
        "selected_model":        {"type": "string"},
        "npv_rank":              {"type": ["integer", "null"]},
        "npv_vs_runner_up_usd":  {"type": ["number",  "null"]},
        "rationale":             {"type": "string"},
    },
    "required": [
        "selection_mode",
        "selected_manufacturer",
        "selected_model",
        "npv_rank",
        "npv_vs_runner_up_usd",
        "rationale",
    ],
}

# ── Top-level schema: two scenarios + battery + brand + evidence ──
PV_RECOMMENDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "optimal": _SCENARIO_SCHEMA,
        "recommended": _SCENARIO_SCHEMA,
        "battery_recommendation": _BATTERY_RECOMMENDATION_SCHEMA,
        "panel_brand_recommendation": _PANEL_BRAND_RECOMMENDATION_SCHEMA,
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["features", "tool_results", "catalog", "user_inputs"],
                    },
                    "quote_or_value": {"type": "string"},
                },
                "required": ["source", "quote_or_value"],
            },
        },
    },
    "required": [
        "optimal",
        "recommended",
        "battery_recommendation",
        "panel_brand_recommendation",
        "evidence",
    ],
}

# Convenient JSON string (for including in prompts)
PV_RECOMMENDATION_SCHEMA_JSON = json.dumps(PV_RECOMMENDATION_SCHEMA, indent=2)


# ── Validation helpers ───────────────────────────────────────

_TYPE_MAP = {
    "integer": int,
    "number": (int, float),
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_SCENARIO_REQUIRED = _SCENARIO_SCHEMA["required"]
_SCENARIO_PROPS = _SCENARIO_SCHEMA["properties"]


def _check_type(value: Any, expected: str) -> bool:
    """Return True if *value* matches the JSON-schema *expected* type."""
    py_type = _TYPE_MAP.get(expected)
    if py_type is None:
        return True  # unknown type → pass
    return isinstance(value, py_type)


def _validate_scenario(data: Dict[str, Any], label: str) -> List[str]:
    """Validate one scenario block (optimal or recommended)."""
    errors: List[str] = []

    for key in _SCENARIO_REQUIRED:
        if key not in data:
            errors.append(f"[{label}] Missing required field: '{key}'")

    for key, spec in _SCENARIO_PROPS.items():
        if key not in data:
            continue
        expected_type = spec.get("type")
        if expected_type and not _check_type(data[key], expected_type):
            errors.append(
                f"[{label}] Field '{key}' expected type '{expected_type}', "
                f"got {type(data[key]).__name__}"
            )

    # Nested objects
    for nested_key in ("constraints", "assumptions"):
        nested_spec = _SCENARIO_PROPS.get(nested_key, {})
        nested_data = data.get(nested_key)
        if nested_data is None:
            continue
        if not isinstance(nested_data, dict):
            errors.append(f"[{label}] '{nested_key}' must be an object")
            continue
        for req in nested_spec.get("required", []):
            if req not in nested_data:
                errors.append(f"[{label}] Missing required field in '{nested_key}': '{req}'")

    # Numeric range checks
    if "confidence" in data and isinstance(data["confidence"], (int, float)):
        if not (0.0 <= data["confidence"] <= 1.0):
            errors.append(f"[{label}] 'confidence' should be between 0 and 1")
    if "target_offset_fraction" in data and isinstance(data["target_offset_fraction"], (int, float)):
        if not (0.0 <= data["target_offset_fraction"] <= 2.0):
            errors.append(f"[{label}] 'target_offset_fraction' looks unreasonable (> 200%)")

    return errors


_BATTERY_REQUIRED = _BATTERY_RECOMMENDATION_SCHEMA["required"]
_BATTERY_PROPS    = _BATTERY_RECOMMENDATION_SCHEMA["properties"]
_VALID_DECISIONS  = {"add_battery", "evaluate_later", "pv_only"}


def _validate_battery_recommendation(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _BATTERY_REQUIRED:
        if key not in data:
            errors.append(f"[battery_recommendation] Missing required field: '{key}'")
    decision = data.get("decision")
    if decision is not None and decision not in _VALID_DECISIONS:
        errors.append(
            f"[battery_recommendation] 'decision' must be one of "
            f"{sorted(_VALID_DECISIONS)}, got '{decision}'"
        )
    for key, spec in _BATTERY_PROPS.items():
        if key not in data:
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        # allow union types like ["number", "null"]
        if isinstance(expected, list):
            if data[key] is not None and not _check_type(data[key], expected[0]):
                errors.append(
                    f"[battery_recommendation] '{key}' expected {expected}, "
                    f"got {type(data[key]).__name__}"
                )
        elif not _check_type(data[key], expected):
            errors.append(
                f"[battery_recommendation] '{key}' expected '{expected}', "
                f"got {type(data[key]).__name__}"
            )
    return errors


_BRAND_REQUIRED  = _PANEL_BRAND_RECOMMENDATION_SCHEMA["required"]
_BRAND_PROPS     = _PANEL_BRAND_RECOMMENDATION_SCHEMA["properties"]
_VALID_BRAND_MODES = {"auto", "user_specified"}


def _validate_panel_brand_recommendation(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _BRAND_REQUIRED:
        if key not in data:
            errors.append(f"[panel_brand_recommendation] Missing required field: '{key}'")
    mode = data.get("selection_mode")
    if mode is not None and mode not in _VALID_BRAND_MODES:
        errors.append(
            f"[panel_brand_recommendation] 'selection_mode' must be one of "
            f"{sorted(_VALID_BRAND_MODES)}, got '{mode}'"
        )
    for key, spec in _BRAND_PROPS.items():
        if key not in data:
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        if isinstance(expected, list):  # union type e.g. ["integer", "null"]
            if data[key] is not None and not _check_type(data[key], expected[0]):
                errors.append(
                    f"[panel_brand_recommendation] '{key}' expected {expected}, "
                    f"got {type(data[key]).__name__}"
                )
        elif not _check_type(data[key], expected):
            errors.append(
                f"[panel_brand_recommendation] '{key}' expected '{expected}', "
                f"got {type(data[key]).__name__}"
            )
    return errors


def validate_recommendation(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate *data* against :data:`PV_RECOMMENDATION_SCHEMA`.

    Returns ``(is_valid, list_of_error_strings)``.
    """
    errors: List[str] = []

    # Top-level required keys
    for key in ("optimal", "recommended", "battery_recommendation",
                "panel_brand_recommendation", "evidence"):
        if key not in data:
            errors.append(f"Missing top-level required field: '{key}'")

    # Validate each PV scenario
    for label in ("optimal", "recommended"):
        scenario = data.get(label)
        if isinstance(scenario, dict):
            errors.extend(_validate_scenario(scenario, label))
        elif label in data:
            errors.append(f"'{label}' must be an object, got {type(data.get(label)).__name__}")

    # Validate battery recommendation
    bat_rec = data.get("battery_recommendation")
    if isinstance(bat_rec, dict):
        errors.extend(_validate_battery_recommendation(bat_rec))
    elif bat_rec is not None:
        errors.append("'battery_recommendation' must be an object")

    # Validate panel brand recommendation
    brand_rec = data.get("panel_brand_recommendation")
    if isinstance(brand_rec, dict):
        errors.extend(_validate_panel_brand_recommendation(brand_rec))
    elif brand_rec is not None:
        errors.append("'panel_brand_recommendation' must be an object")

    # Validate shared evidence array
    evidence = data.get("evidence")
    if isinstance(evidence, list):
        if len(evidence) < 1:
            errors.append("'evidence' array should have at least 1 entry")
        for i, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                errors.append(f"evidence[{i}] must be an object")
                continue
            if "source" not in entry:
                errors.append(f"evidence[{i}] missing 'source'")
            elif entry["source"] not in ("features", "tool_results", "catalog", "user_inputs"):
                errors.append(
                    f"evidence[{i}].source must be one of 'features', "
                    f"'tool_results', 'catalog', 'user_inputs' — got '{entry['source']}'"
                )
            if "quote_or_value" not in entry:
                errors.append(f"evidence[{i}] missing 'quote_or_value'")

    return (len(errors) == 0, errors)


def build_repair_prompt(
    invalid_json: str, errors: List[str], schema_json: Optional[str] = None
) -> str:
    """Create a prompt asking the model to fix its own invalid JSON."""
    schema_block = schema_json or PV_RECOMMENDATION_SCHEMA_JSON
    error_list = "\n".join(f"  - {e}" for e in errors)
    return (
        "Your previous output did not match the required JSON schema.\n\n"
        f"### Errors\n{error_list}\n\n"
        f"### Schema\n```json\n{schema_block}\n```\n\n"
        f"### Your previous output\n```json\n{invalid_json}\n```\n\n"
        "Please output ONLY corrected JSON matching the schema exactly. "
        "No prose, no markdown fences—just the JSON object."
    )
