import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

rng = np.random.default_rng(42)     # reproducibility
n_samples, n_features = 1000, 4

# 1) Draw features from N(0,1)
X = rng.normal(0, 1, size=(n_samples, n_features))

# 2) Define ground-truth weights (feel free to tweak)
w = np.array([2.0, -1.5, 0.5, 1.0])
b = -0.25

# 3) Compute class-1 probabilities with the logistic model
p = expit(X @ w + b)

# 4) Sample binary labels from Bernoulli(p)
y = rng.binomial(1, p)

# 5) Pack into a DataFrame and save
df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
df["y"] = y
import os
os.makedirs("./data", exist_ok=True)
df.to_csv("./data/dummy_logreg_data.csv", index=False)

print(df.head())
