import pytest
from freight_agents.volume_forecast.evaluation import REFERENCE_RUN
from freight_agents.volume_forecast.runs import Run, RunStore


@pytest.fixture(scope="session")
def default_store() -> RunStore:
    """The committed reference run: the default eight-origin run the golden answers describe."""
    return RunStore(REFERENCE_RUN)


@pytest.fixture(scope="session")
def default_run(default_store: RunStore) -> Run:
    return default_store.load("latest")
