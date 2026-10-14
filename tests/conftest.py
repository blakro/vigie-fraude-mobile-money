import pytest

from vigie.data.simulator import simulate


@pytest.fixture(scope="session")
def small():
    """Petite simulation complète (60 jours, 400 clients) partagée par les tests."""
    return simulate(seed=7, n_customers=400, n_agents=40, n_merchants=40)
