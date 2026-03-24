import numpy as np

def fed_avg(weight_list):
    """Federated averaging sederhana"""
    if not weight_list:
        return np.zeros(10)  # fallback default
    return np.mean(weight_list, axis=0)
