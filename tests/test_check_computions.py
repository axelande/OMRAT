import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_load_data import load_data

# Simulate a pick event for the first line
class MockPickEvent:
    def __init__(self, artist):
        self.artist = artist

# Simulate a button press event for the first polygon
class MockButtonEvent:
    def __init__(self, x, y):
        self.xdata = x
        self.ydata = y


def test_calculate_drift(load_data):
    load_data.run_calculation()
    assert load_data.main_widget.result_values.text() == 'Sum of weighted overlaps for this line: 5.822e-02'
    
    a=1