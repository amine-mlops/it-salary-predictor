import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from model import gradient_descent, compute_gradient, compute_cost

df = pd.read_csv("salary_dataset.csv")

x = df[['Experience','YearsInCompany','Certifications','ProjectsCompleted']].values
y = df['Salary'].values

b_init = 8000
w_init = np.array([100, 200, 300, 400])

initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 12000
alpha = 5.0e-3
w_final, b_final, j_hist = gradient_descent(x, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations )
print(f"b,w found by gradient descent:{b_final},{w_final}")
m,_ = x.shape
for i in range(m):
    print(f"prediction: {np.dot(x[i], w_final) + b_final}, target value: {y[i]}")





