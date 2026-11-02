import pytest

from src.demo_scenarios import DEMO_PRODUCTS
from src.simulator import SCENARIOS, compare, run_all_scenarios, scenario_overrides
from src.next_life_engine import analyze_product

LAPTOP = list(DEMO_PRODUCTS.values())[1]


def test_inputs_affect_outputs():
    base = analyze_product(LAPTOP)
    for field, factor, path in (("repair_cost", 0.5, "refurbish"), ("resale_value", 1.5, "resale"),
                                ("component_value", 1.5, "component"), ("logistics_cost", 0.5, "resale")):
        new = analyze_product(LAPTOP, {field: base["baseline_values"][field] * factor})
        assert new["pathways"][path]["net"] != pytest.approx(base["pathways"][path]["net"])


def test_recommendation_flip_is_real():
    r = compare(LAPTOP, {"repair_cost": analyze_product(LAPTOP)["baseline_values"]["repair_cost"] * 0.4})
    assert r["changed"] and r["from_label"] == "Component Recovery" and r["to_label"] == "Refurbish"


def test_damage_severity_and_time():
    base = analyze_product(LAPTOP)
    worse = analyze_product(LAPTOP, {"damage_severity": 1.4})
    assert worse["condition"]["score"] < base["condition"]["score"]
    slow = analyze_product(LAPTOP, {"processing_days": 25})
    assert slow["pathways"]["resale"]["net"] < base["pathways"]["resale"]["net"]


def test_scenarios_run():
    assert len(run_all_scenarios(LAPTOP)) == len(SCENARIOS) + 1
    b = analyze_product(LAPTOP)["baseline_values"]
    assert scenario_overrides("A", b)["repair_cost"] == pytest.approx(b["repair_cost"] * 0.8)
