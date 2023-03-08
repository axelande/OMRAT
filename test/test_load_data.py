import json
import os
import sys

import pytest
from qgis.PyQt.QtWidgets import QApplication

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(os.path.dirname(SCRIPT_DIR))
from open_mrat import OpenMRAT
from utils.gather_data import GatherData
from compute.run_calculations import Calculation


app = QApplication(sys.argv)
def test_gather_data_func():
    omrat = OpenMRAT(None, True)
    omrat.run()
    gd = GatherData(omrat)
    with open('test\\test_res.omrat') as f:
        data = json.load(f)
        gd.populate(data)
    return gd


def test_run_calculation():
    gd = test_gather_data_func()
    data = gd.get_all_for_save()
    calc = Calculation(data, 10)
    exp_drift = {'tot_sum': 0.0006511175081618417, 
                 'l': {'1': {'lin_sum': 0.0006511175081618417, 
                             'North going': 0.0003163098044510794, 
                             'South going': 0.0003348077037107623}}, 
                 'o': {'Structure - 1': 0.00019071317946550977, 
                       'Depth - 1': 0.00046040432869633196}, 
                 'all': {'1 - North going': {'Structure - 1': 9.535177957577136e-05, 
                                             'Depth - 1': 0.00022095802487530805}, 
                         '1 - South going': {'Structure - 1': 9.536139988973841e-05, 
                                             'Depth - 1': 0.0002394463038210239}}}
    exp_power = {'tot_sum': 0.0823688084188536, 
                 'l': {'1': {'lin_sum': 0.0823688084188536, 
                             'North going': 0.0823688084188536, 
                             'South going': 0}}, 
                 'o': {'Structure - 1': 0.0823688084188536, 
                       'Depth - 1': 0}, 
                 'all': {'1 - North going': {'Structure - 1': 0.0823688084188536, 
                                             'Depth - 1': 0}, 
                         '1 - South going': {'Structure - 1': 0, 'Depth - 1': 0}}}
    assert calc.drift_dict == exp_drift
    assert calc.powered_dict == exp_power


if __name__ == '__main__':
    test_gather_data_func()
    test_run_calculation()
    app.quit()