from numpy import exp, log
from scipy  import stats

def get_Fcoll(na:float, pc:float) -> float:
    """Get the accident frequncy """
    return na * pc

def get_drifting_prob(Fb:float, line_length:float, ship_speed:float) -> float:
    """Calculates the drifting probability """
    hours = line_length / ship_speed
    return 1 - exp( - Fb / (24 * 365) * hours)

def get_drift_time(distance:float, drift_speed:float) -> float:
    """Estimates the drifting time """
    return distance / drift_speed

def repairtime_function(data, x) -> float:
    if data["active_window"] == 0:
        drift = stats.lognorm(data["std"], data["loc"], data["scale"])
        repaired = drift.cdf(x)
    else:
        repaired = eval(data["func"])
    return repaired

def powered_na(distance, mean_time, ship_speed):
    ai = mean_time * ship_speed
    return exp(-distance / ai)
    
    

