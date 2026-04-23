import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from pathlib import Path
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
from scipy.signal import argrelmin
import matplotlib.cm as cm





drag = pd.read_csv(Path(__file__).parent / 'drag.csv')
funcion_beta = interp1d(drag["depth"], drag["drag_coeff"], bounds_error=True, fill_value=(1.666, 0.845))

boyancia = 6.857
velocidad_inicial = np.linspace(0,10,60)[1:]
theta = -45
angulos_optimos = []
angulos = np.linspace(-80, -10, 35)
alcances = []

gg = lambda y: funcion_beta(np.clip(y,0.060, 2.540))

def sistema (t, Y):
    x,y,vx,vy = Y
    norma_v = np.sqrt(vx**2 + vy**2)
    if norma_v < 1e-10:
        return np.array([0, 0, 0, boyancia])
    beta = gg(-y)
    return np.array([
        vx,
        vy,
        -beta * norma_v**2.31 * (vx/norma_v),
        -beta * norma_v**2.31 * (vy/norma_v) + boyancia
    ])

def evento(t,Y):
    x,y,vx,vy = Y
    return y

evento.terminal = True
evento.direction = 1

def alcance_negativo(theta, velocidad_inicial):
    theta = float(theta)
    vx0 = velocidad_inicial * np.cos(np.radians(theta))
    vy0 = velocidad_inicial * np.sin(np.radians(theta))
    y0 = np.array([0, 0.0, vx0, vy0], dtype=float)
    sol = solve_ivp(sistema, t_span=(0, 100), y0=y0, max_step=0.01, events=[evento])
    if sol.y_events[0].size > 0:
        return -sol.y_events[0][0][0]
    return 0

angulos_optimos = []
for v0 in velocidad_inicial:
    v0_scalar = float(v0)
    resultado = minimize_scalar(lambda theta: alcance_negativo(theta, v0_scalar), bounds=(-80, -10))
    angulos_optimos.append(resultado.x)

plt.figure(figsize=(12, 5))
plt.plot(velocidad_inicial, angulos_optimos)
plt.xticks(np.arange(0, 10.5, 0.5))
plt.xlabel('v0 (m/s)')
plt.ylabel('Ángulo óptimo (°)')
plt.savefig('1.angle.pdf')





tmax = 250
dt = 0.01
ts = np.arange(0,tmax,dt)
m = 1.7
r_1 = (0,0)
r_2 = (1,1)
v_1 = (0,0.5)
v_2 = (0,-0.5)
G = 1

y0 = np.array([r_1[0], r_1[1], r_2[0], r_2[1], v_1[0], v_1[1], v_2[0], v_2[1]])

def f(t,y):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y

    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)

    F = G * m * m / r**2

    Fx = F * dx/r
    Fy = F * dy/r

    ax1 =  Fx/m
    ay1 =  Fy/m
    ax2 = -Fx/m
    ay2 = -Fy/m

    return np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])

ys = np.empty( (len(ts),8) )
ys[0] = y0

h = dt
for i in range(1,len(ts)):
    t = ts[i-1]
    y = ys[i-1]
    k0 = f(t,y)
    k1 = f(t+h*(1/3), y + h*(1/3*k0))
    k2 = f(t+h*(2/3), y + h*(-1/3*k0 + 1*k1))
    k3 = f(t+h*(1), y + h*(1*k0 -1*k1 +1*k2))
    ys[i] = y + h * (k0*1/8 + k1*3/8 + k2*3/8 + k3*1/8)

def a(pos):
    x1, y1, x2, y2 = pos
    
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    
    F = G * m * m / r**2
    Fx = F * dx/r
    Fy = F * dy/r
    
    return np.array([Fx/m, Fy/m, -Fx/m, -Fy/m])

xs = np.empty((len(ts), 4))
vs = np.empty((len(ts), 4))

xs[0] = y0[:4]
vs[0] = y0[4:]

for i in range(1, len(ts)):
    a_presente = a(xs[i-1])
    xs[i] = xs[i-1] + vs[i-1]*h + a_presente*h**2/2
    a_future = a(xs[i])
    vs[i] = vs[i-1] + (a_presente + a_future)/2 * h

def energia(pos, vel):
    x1,y1,x2,y2 = pos
    vx1,vy1,vx2,vy2 = vel
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    KE = 0.5*m*(vx1**2 + vy1**2 + vx2**2 + vy2**2)
    PE = -G*m*m/r
    return KE + PE

def momento_angular(pos, vel):
    x1,y1,x2,y2 = pos
    vx1,vy1,vx2,vy2 = vel
    L1 = m*(x1*vy1 - y1*vx1)
    L2 = m*(x2*vy2 - y2*vx2)
    return L1 + L2

E_rk = [energia(ys[i,:4], ys[i,4:]) for i in range(len(ts))]
L_rk = [momento_angular(ys[i,:4], ys[i,4:]) for i in range(len(ts))]

E_v = [energia(xs[i], vs[i]) for i in range(len(ts))]
L_v = [momento_angular(xs[i], vs[i]) for i in range(len(ts))]

fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 2, fig)

ax_rungekutta = fig.add_subplot(gs[0, 0])
ax_verlet = fig.add_subplot(gs[0, 1])
ax_energia  = fig.add_subplot(gs[1, :])
ax_momento_angular = fig.add_subplot(gs[2, :])

ax_rungekutta.plot(ys[:,0], ys[:,1], label="estrella 1", c='r')
ax_rungekutta.plot(ys[:,2], ys[:,3], label="estrella 2", c='g')
ax_rungekutta.set_title("Órbitas RK3/8")
ax_rungekutta.legend()

ax_verlet.plot(xs[:,0], xs[:,1], label="estrella 1", lw=0.5)
ax_verlet.plot(xs[:,2], xs[:,3], label="estrella 2", lw=0.5)
ax_verlet.set_title("Órbitas Verlet")
ax_verlet.legend()

ax_energia.plot(ts, E_rk, label="RK3/8")
ax_energia.plot(ts, E_v, label="Verlet")
ax_energia.set_title("Energía")
ax_energia.legend()

ax_momento_angular.plot(ts, L_rk, label="RK3/8")
ax_momento_angular.plot(ts, L_v, label="Verlet")
ax_momento_angular.set_title("Momento Angular")
ax_momento_angular.legend()

plt.tight_layout()
plt.savefig('2.pdf')





a = 0.7
b = 0.8
R = 1
n = 50
tau = np.linspace(0,10, n)[1:]
I = np.linspace(0,2.5, n)[1:]
t_0 = 0
t_max = 100

def neurona (t, Y, tau,I):
    v, w = Y
    dvdt = v - ((v**3)/3) - w + R*I
    dwdt = (v+a-b*w)/tau
    
    return [dvdt,dwdt]

def evento(t,Y,tau,I):
    dv, dw = neurona(t,Y,tau,I)
    norma = dv**2 + dw**2
    return norma - 1e-4

evento.terminal = True

taus, I_ext = np.meshgrid(tau,I,indexing="ij")

A = np.zeros((len(tau), len(I)))
B = np.zeros((len(tau), len(I)))

def simular(tau,I):
    for j in range(len(tau)):
        for i in range(len(I)):
            sol = solve_ivp(neurona, (t_0,t_max), y0=[0.1,0.1], args=(tau[i],I[j]), max_step=0.1, events=evento)

            if len(sol.y_events[0]) <= 0:
                picos_v = np.ptp(sol.y[0])
                picos_w = np.ptp(sol.y[1])
                A[i,j] = picos_v
                B[i,j] = picos_w
    return A, B

funcion_v, funcion_w = simular(tau,I)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
v =ax1.pcolormesh(taus, I_ext, A)
w =ax2.pcolormesh(taus, I_ext, B)
fig.colorbar(w, ax=ax2)
plt.savefig("3.C.pdf")





alpha_vals = np.linspace(0.7, 1.5, 25)
t0, tmax = 10e-3, 10e4 

def state(t,y, alpha): 
    theta, r, P_theta, P_r = y

    dtheta   = P_theta / (r + 1)**2
    dr       = P_r
    dP_theta = -alpha**2 * (r + 1) * np.sin(theta)
    dP_r     = alpha**2 * np.cos(theta) - r + P_theta**2 / (1 + r)**3

    return [dtheta, dr, dP_theta, dP_r]

def evento_theta0(t, y, alpha): 
    return y[0]
evento_theta0.terminal = False 

def simular(alpha): 
    y0 = [np.pi/2,0.0,0.0,0.0]
    t_max = 1000

    sol = solve_ivp(
        state, 
        (0, t_max),
        y0, 
        method="DOP853", 
        args= (alpha,), 
        events = evento_theta0, 
        rtol = 1e-9, 
        atol=1e-10, 
        dense_output=False
    )

    if len(sol.t_events[0]) > 0: 
        r_sec = sol.y_events[0][:,1]
        Pr_sec = sol.y_events[0][:, 3]
    
    else: 
        r_sec, Pr_sec = np.array([]), np.array([])
    
    return alpha, r_sec, Pr_sec

resultados = Parallel(n_jobs=-1)(delayed(simular)(a) for a in alpha_vals)

cmap = plt.cm.viridis
fig, ax = plt.subplots(figsize=(8, 6))
norma = plt.Normalize(alpha_vals.min(), alpha_vals.max())

for alpha, r_sec, Pr_sec in resultados: 
    if len(r_sec) > 0: 
        color=cmap(norma(alpha))
        ax.scatter(r_sec, Pr_sec, s=2, color=color)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norma)
sm.set_array([])

plt.colorbar(sm, ax=ax, label="alpha")
ax.set_xlabel("r")
ax.set_ylabel("Pr")
ax.set_title("Sección de Poincaré: theta = 0")
plt.tight_layout()
plt.savefig('4.pdf')





hbar = 0.1
a    = 0.8
x0   = 10.0
lam  = 1.0 / (a * hbar)

def V(x):
    return (1 - np.exp(a * (x - x0)))**2 - 1

def puntos_de_retorno(eps):
    s  = np.sqrt(eps + 1.0)
    x1 = x0 + np.log(1 - s) / a
    x2 = x0 + np.log(1 + s) / a
    return x1, x2

def schrodinger_primer_orden(x, y, eps):
    return [y[1], (V(x) - eps) / hbar**2 * y[0]]

def shoot(eps):
    x1, x2 = puntos_de_retorno(eps)
    sol = solve_ivp(schrodinger_primer_orden, [x1 - 2.0, x2 + 1.0], [0.0, 1e-6],
                    args=(eps,), max_step=0.01,
                    method='RK45', rtol=1e-6, atol=1e-8)
    return np.hypot(sol.y[0, -1], sol.y[1, -1])

def get_wavefunction(eps, normalize=True):
    x1, x2 = puntos_de_retorno(eps)
    x_start = x1 - 2.0
    x_end   = x2 + 1.0
    sol = solve_ivp(schrodinger_primer_orden, [x_start, x_end], [0.0, 1e-6],
                    args=(eps,), max_step=0.01,
                    method='RK45', rtol=1e-9, atol=1e-11,
                    dense_output=True)
    xs = np.linspace(x_start, x_end, 2000)
    psi = sol.sol(xs)[0]
    if normalize:
        norm = np.trapezoid(psi**2, xs)
        if norm > 0:
            psi /= np.sqrt(norm)
    return xs, psi

eps_scan = np.linspace(-0.9998, -0.003, 600)
norms    = np.array([shoot(e) for e in eps_scan])

idx_min = argrelmin(norms, order=20)[0]

eigenvalues = []
for i in idx_min:
    lo = eps_scan[max(0, i - 25)]
    hi = eps_scan[min(599, i + 25)]
    res = minimize_scalar(shoot, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-12})
    eigenvalues.append(res.x)

eigenvalues = np.sort(eigenvalues)

def eps_theory(n):
    return (2*lam - n - 0.5) * (n + 0.5) / lam**2 - 1

N = len(eigenvalues)
theoretical = np.array([eps_theory(n) for n in range(N)])

with open('5.txt', 'w') as f:
    f.write(f"{'n':>4}  {'eps_numerical':>16}  {'eps_theoretical':>16}  {'pct_diff (%)':>14}\n")
    f.write("-" * 58 + "\n")
    for n, (en, et) in enumerate(zip(eigenvalues, theoretical)):
        pct = abs(en - et) / abs(et) * 100
        f.write(f"{n:>4}  {en:>16.8f}  {et:>16.8f}  {pct:>14.6f}\n")

x1_min, _ = puntos_de_retorno(eigenvalues[0])
x_left  = x1_min - 2.2
x_right = 14.0

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')

x_plot = np.linspace(4, 12, 2000)
vx = V(x_plot)
ax.plot(x_plot, vx, 'k-', lw=2, label='Potencial de Morse', zorder=5)

colors = cm.rainbow(np.linspace(0, 1, N))
scale  = 0.08  

for n, (eps, color) in enumerate(zip(eigenvalues, colors)):
    xs, psi = get_wavefunction(eps)
    psi_plot = psi * scale + eps
    ax.axhline(eps, color=color, lw=0.8, alpha=0.6, linestyle='--')
    ax.plot(xs, psi_plot, color=color, lw=1.2)

ax.set_xlim(x_left, x_right)
ax.set_ylim(-1.15, 0.1)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Energía', fontsize=12)
ax.set_title('Funciones de onda en potencial', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5.plot.pdf', dpi=150, bbox_inches='tight')