"""Built-in demo products. They run through exactly the same engine as any other product."""

DEMO_PRODUCTS = {
    "1 · Smartphone — cracked screen, healthy battery": dict(
        product_name="Smartphone (flagship)", category="Smartphones", original_value=62000, product_age_days=240,
        product_weight=0.2, cosmetic_damage=25, functional_damage=8, missing_accessories=1,
        packaging_condition=80, screen_condition=12, battery_health=91, component_health=92, water_damage=0,
        usage_level=25),
    "2 · Laptop — screen damaged, SSD / RAM / keyboard working": dict(
        product_name="Laptop (14-inch ultrabook)", category="Laptops", original_value=70000, product_age_days=520,
        product_weight=1.9, cosmetic_damage=30, functional_damage=15, missing_accessories=1,
        packaging_condition=70, screen_condition=8, battery_health=82, component_health=90, water_damage=0,
        usage_level=45),
    "3 · Furniture — cosmetic damage, structurally sound": dict(
        product_name="Wooden bookshelf", category="Furniture", original_value=24000, product_age_days=200,
        product_weight=30, cosmetic_damage=40, functional_damage=5, missing_accessories=0,
        packaging_condition=50, component_health=85, water_damage=0, usage_level=20),
    "4 · Clothing — minor defect, good fabric": dict(
        product_name="Cotton jacket", category="Clothing", original_value=3200, product_age_days=120,
        product_weight=0.5, cosmetic_damage=15, functional_damage=10, missing_accessories=0,
        packaging_condition=70, component_health=90, water_damage=0, usage_level=10),
    "5 · Tablet — severe damage, multiple failed components": dict(
        product_name="Tablet (10-inch)", category="Tablets", original_value=38000, product_age_days=500,
        product_weight=0.5, cosmetic_damage=70, functional_damage=85, missing_accessories=2,
        packaging_condition=40, screen_condition=20, battery_health=25, component_health=30, water_damage=1,
        usage_level=60),
}
