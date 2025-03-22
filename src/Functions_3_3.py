import numpy as np

def leapfrog_oscillator(x0, v0, k, m, dt, t_max, F0=0, omega_drive=0):
    t = np.arange(0, t_max, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0] = x0
    v_half = v0 - 0.5 * dt * (-k * x0 / m)
    
    for i in range(len(t) - 1):
        x[i + 1] = x[i] + dt * v_half
        F = -k * x[i + 1] + F0 * np.sin(omega_drive * t[i + 1])
        v_half += dt * (F / m)
        v[i + 1] = v_half + 0.5 * dt * (F / m)
    
    return t, x, v

def harmonic_oscillator(t, y, k, m, F0, omega_drive):
    x, v = y
    return [v, (-k * x + F0 * np.sin(omega_drive * t)) / m]

def leapfrog_energy(x0, v0, k, m, dt, t_max):
    n_steps = int(t_max / dt)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    E = np.zeros(n_steps)
    x[0], v[0] = x0, v0
    omega = np.sqrt(k / m)
    E[0] = 0.5 * m * v0**2 + 0.5 * k * x0**2  
    
    for i in range(1, n_steps):
        v_half = v[i-1] - 0.5 * dt * omega**2 * x[i-1]
        x[i] = x[i-1] + dt * v_half
        v[i] = v_half - 0.5 * dt * omega**2 * x[i]
        E[i] = 0.5 * m * v[i]**2 + 0.5 * k * x[i]**2
    
    return np.linspace(0, t_max, n_steps), x, E



