import pytest

from src.condition_engine import assess_condition
from src.demo_scenarios import DEMO_PRODUCTS
from src.next_life_engine import analyze_product
from src.utils import PATHWAYS

LAPTOP = list(DEMO_PRODUCTS.values())[1]


def test_condition_score_and_grade():
    c = assess_condition(LAPTOP)
    assert 0 <= c["score"] <= 100 and c["grade"] in "ABCDE"
    pristine = assess_condition(dict(category="Laptops", original_value=50000))
    assert pristine["grade"] == "A" and pristine["score"] > 95


@pytest.mark.parametrize("name", list(DEMO_PRODUCTS))
def test_net_recovery_arithmetic(name):
    a = analyze_product(DEMO_PRODUCTS[name])
    for k in PATHWAYS:
        p = a["pathways"][k]
        assert p["net"] == pytest.approx(p["gross"] - sum(p["costs"].values()))
        assert p["gross"] >= 0 and all(v >= 0 for v in p["costs"].values())
        assert p["value_preserved"] == pytest.approx(p["net"] / a["record"]["original_value"] * 100)


def test_disposal_has_no_gross_and_negative_net():
    d = analyze_product(LAPTOP)["pathways"]["disposal"]
    assert d["gross"] == 0 and d["net"] < 0


def test_pristine_item_is_resold_and_wrecked_item_is_not():
    good = analyze_product(dict(category="Smartphones", original_value=30000, product_age_days=60))
    assert good["recommended"] == "resale"
    wreck = analyze_product(list(DEMO_PRODUCTS.values())[4])
    assert wreck["recommended"] in ("component", "material", "disposal")


def test_invalid_input_raises_friendly_error():
    with pytest.raises(ValueError):
        analyze_product(dict(category="Spaceships", original_value=10))
    with pytest.raises(ValueError):
        analyze_product(dict(category="Laptops", original_value=-1))


def test_categories_have_different_economics():
    ph = analyze_product(dict(category="Smartphones", original_value=20000, functional_damage=60, component_health=70))
    fu = analyze_product(dict(category="Furniture", original_value=20000, functional_damage=60, component_health=70))
    assert ph["pathways"]["component"]["gross"] != fu["pathways"]["component"]["gross"]
