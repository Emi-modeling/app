import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# -------------------------------
# Load calibration data
# -------------------------------
def load_calibration_data(path="data/All_Data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required_cols = ["B", "CON", "NE"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV must contain column: {col}. Found columns: {df.columns.tolist()}")
    B = (df["B"].values[1:] - 1)/6
    PNB = (df["CON"].values[1:] - 1)/5
    NE = (df["NE"].values[1:] - 1)/5
    data = np.column_stack([B, PNB, NE])
    mu = np.mean(data, axis=0)
    Sigma = np.cov(data, rowvar=False)
    return B, PNB, NE, mu, Sigma

# -------------------------------
# ADN2l and ABM
# -------------------------------
def ADN2l_fast(n, m1, m2, a=1.0, x=None, mu=0.5):
    W1 = np.zeros((n,n))
    W2 = np.zeros((n,n))
    m1 = np.full(n, m1) if np.isscalar(m1) else np.array(m1)
    m2 = np.full(n, m2) if np.isscalar(m2) else np.array(m2)
    p_r = 1/(1+mu) if x is not None else None
    for i in range(n):
        idx1 = np.random.choice(n, m1[i], replace=False)
        W1[i, idx1] = 1/m1[i]
        done = False
        while not done:
            idx2 = np.random.choice(n, m2[i], replace=False)
            if x is None or np.random.rand() < p_r**(m2[i]-np.sum(x[idx2])):
                done = True
        W2[i, idx2] = 1/m2[i]
    return W1, W2

def ABM_fast(n, md, mo, beta, gamma, lambda_, alpha, epsilon, rev, x0, z0, y0, mu_v, T):
    x = np.zeros((n,T))
    y = np.zeros((n,T))
    z = np.zeros((n,T))
    x[:,0], y[:,0], z[:,0] = x0, y0, z0
    beta = np.full(n, beta) if np.isscalar(beta) else np.array(beta)
    gamma = np.full(n, gamma) if np.isscalar(gamma) else np.array(gamma)
    lambda_ = np.full(n, lambda_) if np.isscalar(lambda_) else np.array(lambda_)
    alpha = np.full(n, alpha) if np.isscalar(alpha) else np.array(alpha)
    epsilon = np.full(n, epsilon) if np.isscalar(epsilon) else np.array(epsilon)
    for t in range(1,T):
        W1, W2 = ADN2l_fast(n, md, mo, a=1.0, x=x[:,t-1], mu=mu_v)
        y[:,t] = (1-beta)*y[:,t-1] + beta*gamma*(W1 @ z[:,t-1]) + beta*(1-gamma)*(W2 @ x[:,t-1])
        xtemp = (1-lambda_)*z[:,t-1] + lambda_*y[:,t-1]
        flag_eps = np.random.rand(n) < epsilon
        xtemp_vec = flag_eps*np.round(np.random.rand(n)) + (1-flag_eps)*(xtemp >= alpha)
        flag_rev = np.random.rand(n) < rev
        x[:,t] = flag_rev*xtemp_vec + (1-flag_rev)*x[:,t-1]
        z[:,t] = z[:,t-1] + 0.01*(np.mean(y[:,t-1]) - z[:,t-1])
    return x, y, z

# -------------------------------
# Streamlit app
# -------------------------------
st.title("Transport Simulation ABM")

rad_km_2026 = st.slider("Bike radius 2026 (km)", 50, 200, 100)
beta_slider = st.slider("Beta adaptation", 0, 100, 50)
alpha_sensitivity = st.slider("Alpha sensitivity", 0.0, 1.0, 0.3)

B, PNB, NE, mu, Sigma = load_calibration_data()

n, T = 200, 500
beta = 0.5 * beta_slider / 100
alpha_old = 1 - 90.4/1000
alpha_new = 1 - rad_km_2026/1000
change = alpha_new / alpha_old
alpha_change = change * alpha_new
alpha_norm = alpha_change / (alpha_change + 1)
alpha = (1-alpha_sensitivity)*alpha_new + alpha_sensitivity*alpha_norm

raw = multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
init = np.clip(raw, 0, 1)
init[:,0] = (np.random.rand(n) < 0.5).astype(float)

x, y, z = ABM_fast(n, 10, 10, beta, 0.5, 0.1, alpha, 0.01, 0.1, init[:,0], init[:,2], init[:,1], 0.5, T)
share_bike = np.sum(x[:,-1])/n
st.write(f"Share of bike users: {share_bike:.2f}")
