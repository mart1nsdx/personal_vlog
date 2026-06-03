import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import corner
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba import njit
random = np.random.default_rng()
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy import stats





two_hc  = 39728.91714  
hc_kB   = 1.438776878  


data = np.loadtxt('./CMB_dipole_spectrum.dat', skiprows=2)
nu   = data[:, 0]  
F    = data[:, 1] 
sigma = data[:, 3]


def planck(nu, T, mu=0):
    x = hc_kB * nu / T
    return two_hc * nu**3 / (np.exp(x + mu) - 1)

def dB_dT(nu, T):
    x = hc_kB * nu / T
    ex = np.exp(x)
    return two_hc * nu**3 * (hc_kB / T**2) * ex * x / (ex - 1)**2

def F_model(nu, T_amp, T_CMB, T_Gal, G):
    return T_amp * dB_dT(nu, T_CMB) + G * planck(nu, T_Gal, mu=-1)


def model_wrap(nu, T_amp, T_CMB, T_Gal, G):
    return F_model(nu, T_amp, T_CMB, T_Gal, G)

p0 = [3.41e-3, 2.72, 13.5, 0.425e-8]
popt, pcov = curve_fit(model_wrap, nu, F, p0=p0, sigma=sigma, maxfev=10000)


def log_likelihood(params):
    T_amp, T_CMB, T_Gal, G = params
    Fm = F_model(nu, T_amp, T_CMB, T_Gal, G)
    return -0.5 * np.sum(((Fm - F) / sigma)**2 + np.log(2 * np.pi * sigma**2))

def log_prior(params):
    T_amp, T_CMB, T_Gal, G = params
    if T_CMB <= 0 or T_Gal <= 0 or T_amp <= 0 or G <= 0:
        return -np.inf
    return 0.0

def log_f(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood(params) + lp


M = 50_000
sigma_perturb = np.abs(popt) * 0.001

muestras  = np.zeros((M, 4))
f_values  = np.zeros(M)

muestras[-1] = popt   
f_values[-1] = log_f(muestras[-1])


n_accepted = 0
for i in tqdm(range(M), miniters=1000):
    old       = muestras[i-1]
    new       = old + random.normal(loc=0.0, scale=sigma_perturb)
    log_f_old = f_values[i-1]
    log_f_new = log_f(new)

    log_u = np.log(random.random())
    if log_u < log_f_new - log_f_old:
        muestras[i] = new
        f_values[i] = log_f_new
        n_accepted += 1
    else:
        muestras[i] = old
        f_values[i] = log_f_old


burn_in = 5000
muestras_post = muestras[burn_in:]

fig = corner.corner(
    muestras_post,
    labels=[r'$T_{\rm amp}$ (K)', r'$T_{\rm CMB}$ (K)', r'$T_{\rm Gal}$ (K)', r'$G$'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 11}
)

fig.savefig('1.pdf')





N = 50
J = 1.0
beta = 0.5
epocas = 200_000


@njit
def energia_iman(grid, i, j, J, N):
    vecinos = (grid[(i+1) % N, j] + grid[(i-1) % N, j] +
               grid[i, (j+1) % N] + grid[i, (j-1) % N])
    return -J * grid[i, j] * vecinos

@njit
def energia_total(grid, J, N):
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += energia_iman(grid, i, j, J, N)
    return E


@njit
def simular_ising(grid, J, beta, N, epocas):
    E_hist = np.zeros(epocas)
    M_hist = np.zeros(epocas)

    E = energia_total(grid, J, N)
    M = grid.sum()

    for k in range(epocas):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        eps_old = energia_iman(grid, i, j, J, N)
        grid[i, j] *= -1
        eps_new = energia_iman(grid, i, j, J, N)
        delta_eps = eps_new - eps_old

        if delta_eps <= 0 or np.random.random() < np.exp(-beta * delta_eps):
            E += delta_eps
            M += 2 * grid[i, j]
        else:
            grid[i, j] *= -1
            delta_eps = 0.0

        E_hist[k] = E / (4 * N**2)
        M_hist[k] = M / N**2

    return E_hist, M_hist, grid


grid_inicial = np.random.choice([-1, 1], size=(N, N)).astype(np.int64)
grid_final   = grid_inicial.copy()

E_hist, M_hist, grid_final = simular_ising(grid_final, J, beta, N, epocas)


fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(grid_inicial, cmap='RdYlGn', vmin=-1, vmax=1)
ax1.set_title('Antes')
ax1.axis('off')

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title('Durante')
ax2.set_xlabel('Épocas')

for sim in range(10):
    grid = np.random.choice([-1, 1], size=(N, N)).astype(np.int64)
    E_hist, M_hist, grid_final = simular_ising(grid.copy(), J, beta, N, epocas)
    ax2.plot(np.arange(epocas), E_hist, c='k', lw=0.5, alpha=0.5)
    ax2.plot(np.arange(epocas), M_hist, c='r', lw=0.5, alpha=0.4)


ax2.legend(handles=[
    Line2D([0], [0], color='k', label='Energía'),
    Line2D([0], [0], color='r', label='Magnetización')
])

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(grid_final, cmap='RdYlGn', vmin=-1, vmax=1)
ax3.set_title('Después')
ax3.axis('off')

plt.tight_layout()
plt.savefig('2.a.pdf')






A = 1000
B= 20
t_medios_U = 23.4 / 1140
t_medios_Np = 2.36
lambda_U = np.log(2)/t_medios_U
lambda_Np = np.log(2)/t_medios_Np

dU_estable = A/lambda_U
dNP_estable = (lambda_U*dU_estable)/lambda_Np
dPU_estable = (lambda_Np*dNP_estable)/B


def sistema(t, Y):
    U, Np, Pu = Y
    dUdt = A - lambda_U * U
    dNpdt = lambda_U * U - lambda_Np * Np
    dPudt = lambda_Np * Np - B * Pu
    return [dUdt, dNpdt, dPudt]


def eventos(t, Y):
    dUdt, dNpdt, dPudt = sistema(t, Y)
    return np.sqrt(dUdt**2 + dNpdt**2 + dPudt**2) - 1e-3

eventos.terminal = False

y0 = np.array([10, 10, 10])
t_span = (0, 30)
t_eval = np.linspace(0, 30, 3000)

sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval, events=eventos, max_step=0.01)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = [r'$U^{239}_{92}$', r'$Np^{239}_{93}$', r'$Pu^{239}_{94}$']
ss_values = [dU_estable, dNP_estable, dPU_estable]
colors = ['steelblue', 'darkorange', 'forestgreen']

for i, (ax, label, ss, color) in enumerate(zip(axes, labels, ss_values, colors)):
    ax.plot(sol.t, sol.y[i], color=color, label=label)
    ax.axhline(ss, color="k", linestyle='--', alpha=0.7, label=f'Estado estable: {ss:.2f}')
    ax.set_ylabel('Cantidad')
    ax.legend(loc='right')
    ax.set_yscale('log')

axes[-1].set_xlabel('Tiempo (días)')
fig.suptitle('Aproximación al promedio')
plt.tight_layout()
plt.savefig('3.a.pdf')


@njit
def sistema_njit(t, Y, A, lambda_U, lambda_Np, B):
    U, Np, Pu = Y
    dUdt = A - lambda_U * U
    dNpdt = lambda_U * U - lambda_Np * Np
    dPudt = lambda_Np * Np - B * Pu
    return np.array([dUdt, dNpdt, dPudt])

@njit
def drift(Y, A, lambda_U, lambda_Np, B):
    return sistema_njit(0, Y, A, lambda_U, lambda_Np, B)

@njit
def volatilidad(Y, A, lambda_U, lambda_Np, B):
    U, Np, Pu = Y
    sigma_U = np.sqrt(A + lambda_U * abs(U))
    sigma_Np = np.sqrt(lambda_U * abs(U) + lambda_Np * abs(Np))
    sigma_Pu = np.sqrt(lambda_Np * abs(Np) + B * abs(Pu))
    return np.array([sigma_U, sigma_Np, sigma_Pu])

@njit
def runge_kutta(Y, dt, A, lambda_U, lambda_Np, B):
    W = np.random.normal(0.0, 1.0, 3)
    S = np.sign(np.random.random(3) - 0.5)
    K1 = dt * drift(Y, A, lambda_U, lambda_Np, B) + (W - S) * np.sqrt(dt) * volatilidad(Y, A, lambda_U, lambda_Np, B)
    y1 = Y + K1
    K2 = dt * drift(y1, A, lambda_U, lambda_Np, B) + (W + S) * np.sqrt(dt) * volatilidad(y1, A, lambda_U, lambda_Np, B)
    return Y + 0.5 * (K1 + K2)

dt = 0.001
ts = np.arange(0, 30, dt)
n_traj = 5
ys = np.empty((len(ts), 3))
ys[0] = y0

trajectories_b = []
for _ in range(n_traj):
    ys = np.empty((len(ts), 3))
    ys[0] = y0
    for i in range(1, len(ts)):
        ys[i] = runge_kutta(ys[i-1], dt, A, lambda_U, lambda_Np, B)
        ys[i] = np.maximum(ys[i], 0)
    trajectories_b.append(ys)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = [r'$U^{239}_{92}$', r'$Np^{239}_{93}$', r'$Pu^{239}_{94}$']
colors = ['steelblue', 'darkorange', 'forestgreen']
traj_colors = ['red', 'blue', 'purple', 'orange', 'green']

for i, (ax, label) in enumerate(zip(axes, labels)):
    for j, traj in enumerate(trajectories_b):
        ax.plot(ts, traj[:, i], color=traj_colors[j], alpha=0.4, linewidth=0.5)
    ax.plot(sol.t, sol.y[i], color='black', linewidth=0.5, linestyle='--', label=f'Estado estable {label}')
    ax.set_ylabel('Cantidad')
    ax.set_yscale('log')
    ax.legend(loc='right')

axes[-1].set_xlabel('Tiempo (días)')
fig.suptitle("Runge-Kutta")
plt.tight_layout()
plt.savefig('3.b.pdf')


@njit
def ttasas(U, Np, Pu):
    return np.array([A, U * lambda_U, Np * lambda_Np, Pu * B])

@njit
def ssa_step(U, Np, Pu):
    tasas = ttasas(U, Np, Pu)
    u = np.random.rand()
    k = np.searchsorted(np.cumsum(tasas / tasas.sum()), u)
    tau = np.random.exponential(1 / tasas.sum())
    if k == 0:
        U += 1
    elif k == 1:
        U -= 1; Np += 1
    elif k == 2:
        Np -= 1; Pu += 1
    elif k == 3:
        Pu -= 1
    return tau, U, Np, Pu

@njit
def gillespie(U0, Np0, Pu0, t_max):
    U, Np, Pu = U0, Np0, Pu0
    t_old = 0.0
    ts = [t_old]
    Us, Nps, Pus = [U], [Np], [Pu]
    while t_old < t_max:
        tau, U_new, Np_new, Pu_new = ssa_step(U, Np, Pu)
        t_new = t_old + tau
        if t_new > t_max:
            break
        U, Np, Pu = max(U_new, 0.0), max(Np_new, 0.0), max(Pu_new, 0.0)
        t_old = t_new
        ts.append(t_old)
        Us.append(U)
        Nps.append(Np)
        Pus.append(Pu)
        
    return np.array(ts), np.array(Us), np.array(Nps), np.array(Pus)

trajectories_c = []
for _ in range(n_traj):
    ts_g, Us, Nps, Pus = gillespie(y0[0], y0[1],y0[2], 30.0)
    ys_g = np.column_stack((Us, Nps, Pus))
    trajectories_c.append((ts_g, ys_g))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = [r'$U^{239}_{92}$', r'$Np^{239}_{93}$', r'$Pu^{239}_{94}$']
colors = ['steelblue', 'darkorange', 'forestgreen']

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    for j, (ts_g, ys_g) in enumerate(trajectories_c):
        lbl = label if j == 0 else None
        ax.plot(ts_g, ys_g[:, i], color=colors[i], alpha=0.3, linewidth=0.8, label=lbl)
    ax.plot(sol.t, sol.y[i], color='black', linewidth=1.5, linestyle='--', label='Determinista')
    ax.set_ylabel('Cantidad')
    ax.set_yscale('log')
    ax.legend(loc='right')

axes[-1].set_xlabel('Tiempo (días)')
fig.suptitle('Simulación exacta')
plt.tight_layout()
plt.savefig('3.c.pdf')


N = 1000
k_g = 0
for _ in range(N):
    ts_g, Us, Nps, Pus = gillespie(y0[0], y0[1], y0[2], 30.0)
    if np.max(Pus) >= 80:
        k_g += 1

dist_g = stats.beta(a=1+k_g, b=1+N-k_g)
samples_g = dist_g.rvs(10000)

xmin_g, xmid_g, xmax_g = np.quantile(samples_g, [0.16, 0.50, 0.84])
resultado_g = f"${100*xmid_g:.2f}^{{+{100*(xmax_g-xmid_g):.2f}}}_{{-{100*(xmid_g-xmin_g):.2f}}}$ %"

plt.hist(samples_g, bins=100, density=True)
plt.axvline(xmid_g, c='k')
plt.title(resultado_g)
plt.xlabel('p')
plt.tight_layout()
plt.savefig('3.d.Gillespie.pdf')

k_rk = 0
dt= 0.01
ts=np.arange(0,30,dt)

for _ in range(N):
    ys_rk = np.empty((len(ts), 3))
    ys_rk[0] = y0
    for i in range(1, len(ts)):
        ys_rk[i] = runge_kutta(ys_rk[i-1], dt, A, lambda_U, lambda_Np, B)
        ys_rk[i] = np.maximum(ys_rk[i], 0)
    if np.max(ys_rk[:, 2]) >= 80:
        k_rk += 1

dist_rk = stats.beta(a=1+k_rk, b=1+N-k_rk)
samples_rk = dist_rk.rvs(10000)
xmin_rk, xmid_rk, xmax_rk = np.quantile(samples_rk, [0.16, 0.50, 0.84])
resultado_rk = f"${100*xmid_rk:.2f}^{{+{100*(xmax_rk-xmid_rk):.2f}}}_{{-{100*(xmid_rk-xmin_rk):.2f}}}$ %"

plt.hist(samples_rk, bins=100, density=True)
plt.axvline(xmid_rk, c='k')
plt.title(resultado_rk)
plt.xlabel('p')
plt.tight_layout()
plt.savefig('3.d.Runge-Kutta.pdf')

with open('3.d.txt', 'w') as f:
    f.write(f"""Al hacer la probabilidad de llegar a la concentración crítica para cada método, con Gillespie se obtuvo {resultado_g}, mientras
            que con Runge-Kutta fue {resultado_rk}. Estas probabilidades tienen sentido respecto a su método ya que el Runge-Kutta (de segundo 
            orden) hace una aproximación sutil al ser una simluación de ecuaciones diferenciales, mientras que Gillespie hace un análisis 
            estadístico mejor del sistema al ser un método estadístico""")