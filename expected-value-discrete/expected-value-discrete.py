import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if len(x) != len(p):
        raise ValueError("x and p must have the same length")
    if not np.isclose(np.sum(p), 1.0, atol = 1e-6):
        raise ValueError("Probabilities must sum to 1")
    return float(np.sum(np.array(x)*np.array(p)))
    
