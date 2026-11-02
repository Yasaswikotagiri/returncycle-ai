import pytest

from src.data_generator import build


@pytest.fixture(scope="session")
def small_data():
    """Tiny in-memory synthetic dataset (nothing is written to disk)."""
    return build("demo", orders_override=1500, write=False)
