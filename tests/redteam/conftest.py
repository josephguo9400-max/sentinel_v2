import pytest

from redteam.scenario import find_scenario
from redteam.trace import RunTrace


@pytest.fixture
def trace():
    return RunTrace(out_dir=None)


@pytest.fixture
def digest():
    return find_scenario("dev_daily_digest")


@pytest.fixture
def manager():
    return find_scenario("dev_send_summary_to_manager")


@pytest.fixture
def cc_team():
    return find_scenario("ho_cc_team_update")
