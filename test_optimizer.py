from src.demo_scenarios import DEMO_PRODUCTS
from src.next_life_engine import analyze_product

LAPTOP = list(DEMO_PRODUCTS.values())[1]


def test_recommended_has_highest_score_among_eligible():
    a = analyze_product(LAPTOP)
    best = max((a["pathways"][k] for k in a["eligible"]), key=lambda p: p["score"])
    assert best["key"] == a["recommended"]
    assert 0.3 <= a["confidence"] <= 0.97


def test_scores_normalised():
    a = analyze_product(LAPTOP)
    assert all(0 <= p["score"] <= 100 for p in a["pathways"].values())


def test_changing_costs_changes_recommendation():
    a = analyze_product(LAPTOP)
    assert a["recommended"] == "component"
    cheap = analyze_product(LAPTOP, {"repair_cost": a["baseline_values"]["repair_cost"] * 0.4})
    assert cheap["recommended"] == "refurbish"
    pricey = analyze_product(LAPTOP, {"component_value": 0})
    assert pricey["recommended"] != "component"


def test_disposal_only_when_alternatives_lose_more():
    a = analyze_product(LAPTOP)
    assert "disposal" not in a["eligible"]
    nothing = analyze_product(dict(category="Clothing", original_value=500, functional_damage=100, cosmetic_damage=100,
                                   component_health=0, water_damage=1))
    assert nothing["recommended"] == "disposal" or nothing["pathways"][nothing["recommended"]]["net"] >= nothing["pathways"]["disposal"]["net"]
