import numpy as np

def linear_regression_closed_form(X, y):
    X = np.array(X)
    y = np.array(y)

    X_transpose = X.T

    xtx = X_transpose @ X 

    xtx_inverse = np.linalg.inv(xtx)

    xty = X_transpose @ y
    
    w = xtx_inverse @ xty

    return w
    pass