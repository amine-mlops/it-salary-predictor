import copy, math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def predict(x, w, b):
    f = np.dot(x, w) + b
    return f

def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = predict(x[i], w, b)
        cost += (f_wb - y[i])**2
    cost = cost / (2*m)
    return cost

def compute_gradient(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = predict(x[i], w, b) - y[i]
        
        dj_dw = dj_dw + err * x[i]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, compute_cost, compute_gradient, alpha, num_iters):
    j_store = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        if i < 1000:
            j_store.append(compute_cost(x, y, w, b))
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {j_store[-1]} ")
    return w, b, j_store

















