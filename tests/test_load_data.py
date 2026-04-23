import pytest

from omrat import OMRAT
from tests.conftest import omrat


@pytest.fixture()
def load_data(omrat:OMRAT):
    omrat.run()
    omrat.load_work()
    yield omrat
    
def test_the_loaded_data(load_data):
    assert load_data.main_widget.twObjectList.rowCount() == 1
    assert load_data.main_widget.twDepthList.rowCount() == 1
    assert load_data.main_widget.twRouteList.rowCount() == 3
