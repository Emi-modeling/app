import numpy as np


n = 500
T = 500
md = 10
mo = 10
epsilon = 0.01
rev = 0.1
mu_v = 0.8
gamma = 0.5
lam = 0.1  # lambda in Python ist reserviert, deshalb 'lam'
rad_km_2023 = 90.4

# --- Set data ---
mu_B = 0.01
mu_PNB = 0.6029
mu_NE = 0.4412

mu = np.array([mu_B, mu_PNB, mu_NE])

# Initial behaviors (x0) with 1% randomly set to 1
x0 = np.zeros(n)
num_infected = round(0.01 * n)
x0[np.random.permutation(n)[:num_infected]] = 1

# Convictions
z = 0.8 * np.ones(n)

# Initial normative expectations
# y0 = 0.4412 * np.ones(n)  # optional alternative
y0 = 0.8 * np.ones(n)

def ADN2l(n, m1, m2, a=None, x=None, mu=None):

    # Convert m1, m2 to vectors if scalars
    if np.isscalar(m1):
        m1 = np.full(n, m1, dtype=int)
    else:
        m1 = np.array(m1, dtype=int)

    if np.isscalar(m2):
        m2 = np.full(n, m2, dtype=int)
    else:
        m2 = np.array(m2, dtype=int)

    # Empty networks
    W1 = np.zeros((n, n))
    W2 = np.zeros((n, n))

    # Case 1: With activation probability
    if a is not None and x is None and mu is None:
        flag = np.random.rand(n) < a
        for i in range(n):
            if flag[i]:
                targets1 = np.random.permutation(n)[:m1[i]]
                W1[i, targets1] = 1.0 / m1[i]

                targets2 = np.random.permutation(n)[:m2[i]]
                W2[i, targets2] = 1.0 / m2[i]

    # Case 2: Without activation probability
    elif a is None and x is None and mu is None:
        for i in range(n):
            targets1 = np.random.permutation(n)[:m1[i]]
            W1[i, targets1] = 1.0 / m1[i]

            targets2 = np.random.permutation(n)[:m2[i]]
            W2[i, targets2] = 1.0 / m2[i]

    # Case 3: With attractiveness bias
    elif a is not None and x is not None and mu is not None:
        x = np.array(x)
        p_r = 1.0 / (1.0 + mu)

        for i in range(n):
            # Layer 1 (uniform)
            targets1 = np.random.permutation(n)[:m1[i]]
            W1[i, targets1] = 1.0 / m1[i]

            # Layer 2 (biased)
            flag = False
            while not flag:
                targets2 = np.random.permutation(n)[:m2[i]]
                exponent = m2[i] - np.sum(x[targets2])
                if np.random.rand() < p_r ** exponent:
                    W2[i, targets2] = 1.0 / m2[i]
                    flag = True

    else:
        raise ValueError("Invalid combination of arguments.")

    return W1, W2


def ABM(n, md, mo, beta, gamma, lam, alpha, epsilon,
        rev, x0, z, y0, mu, T):

    # --- Convert everything to numpy arrays ---
    def to_vector(param):
        if np.isscalar(param):
            return np.full(n, param, dtype=float)
        return np.array(param, dtype=float)

    beta = to_vector(beta)
    gamma = to_vector(gamma)
    lam = to_vector(lam)
    alpha = to_vector(alpha)
    epsilon = to_vector(epsilon)
    md = to_vector(md).astype(int)
    mo = to_vector(mo).astype(int)

    z = np.array(z, dtype=float)
    x0 = np.array(x0, dtype=float)
    y0 = np.array(y0, dtype=float)

    # --- Initialization ---
    x = np.zeros((n, T))
    y = np.zeros((n, T))

    x[:, 0] = x0
    y[:, 0] = y0

    # --- Time loop ---
    for t in range(1, T):

        # Generate dynamic 2-layer network
        W1, W2 = ADN2l(n, md, mo, a=1, x=x[:, t-1], mu=mu)

        # Update normative expectations
        y[:, t] = (
            (1 - beta) * y[:, t-1]
            + beta * gamma * (W1 @ z)
            + beta * (1 - gamma) * (W2 @ x[:, t-1])
        )

        # Compute threshold (IMPORTANT: uses y(t-1), as in MATLAB)
        xtemp = (1 - lam) * z + lam * y[:, t-1]

        # Bounded rationality mechanism
        flag_random = np.random.rand(n) < epsilon
        random_choices = np.round(np.random.rand(n))

        xtemp = (
            flag_random * random_choices
            + (1 - flag_random) * (xtemp >= alpha)
        )

        # Revision decision
        flag_revision = np.random.rand(n) < rev

        x[:, t] = (
            flag_revision * xtemp
            + (1 - flag_revision) * x[:, t-1]
        )

    return x, y, z
