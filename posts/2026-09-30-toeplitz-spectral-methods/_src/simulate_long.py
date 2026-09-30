import numpy as np, time
from scipy.integrate import solve_ivp
def duffing(alpha, beta, gamma, delta, omega):
    def f(t, z):
        x, y = z
        return [y, -delta * y - alpha * x - beta * x ** 3 + gamma * np.cos(omega * t)]
    return f
t0 = time.time()
dt, T = 0.1, 80000.0
t_eval = np.arange(0, T, dt)
sol = solve_ivp(duffing(-1, 1, 0.5, 0.3, 1.0), (0, T), [0.0, 0.0], t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-11)
X = sol.y.T[5000:]
np.save("chaotic_long.npy", X.astype(np.float32))
print(X.shape, time.time() - t0)
sol = solve_ivp(duffing(0.5, 0.625, 2, 1.5, 1.0), (0, 400), [0.0, 0.0], t_eval=np.arange(0, 400, 0.1), method="DOP853", rtol=1e-9, atol=1e-11)
np.save("periodic_clean.npy", sol.y.T[1000:].astype(np.float32))
