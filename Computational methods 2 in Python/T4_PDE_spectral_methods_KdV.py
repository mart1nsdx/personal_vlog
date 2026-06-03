import numpy as np
import pde
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit





L     = 2 * np.pi  
N     = 512             
delta = 0.022   
dx = L/N        
x = np.linspace(0, L, N, endpoint=False)

phi0= np.exp(-(x-np.pi)**2)

k = np.fft.rfftfreq(N, d=dx)*2*np.pi

ik3 = -1j*k**3

def deriv1(f): 
    return np.fft.irfft(1j*k*np.fft.rfft(f), n=N)

def deriv3(f):
    return np.fft.irfft(ik3 * np.fft.rfft(f), n=N)

def rhs(phi):
    return -(phi * deriv1(phi) + delta**2 * deriv3(phi))

def rk4(phi, dt):
    k1 = rhs(phi)
    k2 = rhs(phi + 0.5*dt*k1)
    k3 = rhs(phi + 0.5*dt*k2)
    k4 = rhs(phi + dt*k3)
    return phi + dt/6*(k1 + 2*k2 + 2*k3 + k4)

dt = 1e-4
t_end = 30.0
n_steps = int(t_end / dt)
n_save  = max(1, n_steps // 800) 

phi       = phi0.copy()
phi_hist  = []
t_hist    = []
mass_hist = []
mom_hist  = []
eng_hist  = []

for step in range(n_steps):
    phi = rk4(phi, dt)
    t   = (step + 1) * dt

    dphi = deriv1(phi)
    mass = np.sum(phi)*dx
    mom  = np.sum(phi**2)* dx
    eng  = 0.5 * np.sum(phi**3/3 - (delta*dphi)**2)*dx

    mass_hist.append(mass)
    mom_hist.append(mom)
    eng_hist.append(eng)
    t_hist.append(t)

    if step % n_save == 0:
        phi_hist.append(phi.copy())

phi_hist  = np.array(phi_hist)
t_anim    = np.array(t_hist[::n_save])
t_hist    = np.array(t_hist)
mass_hist = np.array(mass_hist)
mom_hist  = np.array(mom_hist)
eng_hist  = np.array(eng_hist)

fig_a, ax_a = plt.subplots(figsize=(10, 4))
line_a, = ax_a.plot(x, phi_hist[0], 'b-', lw=1.5)
ax_a.set_xlim(0, L)
ax_a.set_ylim(-0.3, 1.1)
ax_a.set_xlabel('x')
ax_a.set_ylabel('phi(x, t)')
ax_a.set_title('KdV solitons')
ax_a.grid(True, alpha=0.3)
time_txt = ax_a.text(0.02, 0.93, '', transform=ax_a.transAxes)

def animate_a(i):
    line_a.set_ydata(phi_hist[i])
    time_txt.set_text(f't = {t_anim[i]:.2f}')
    return line_a, time_txt

ani_a = FuncAnimation(fig_a, animate_a, frames=len(phi_hist), interval=40, blit=True)
ani_a.save('3.a.mp4', writer=FFMpegWriter(fps=24, bitrate=2000))
plt.close(fig_a)

n_detected = np.array([
    len(find_peaks(f, height=0.05, distance=N//30)[0])
    for f in phi_hist
])
i_best    = np.argmax(n_detected)
phi_sep   = phi_hist[i_best]
peaks0, _ = find_peaks(phi_sep, height=0.05, distance=N//30)

WINDOW = 15   

def track_soliton(i_start, pk_start, n_frames=200):
    cur       = pk_start
    positions = [x[pk_start]]
    times     = [t_anim[i_start]]
    local_h   = [phi_hist[i_start][pk_start]]

    for i in range(i_start + 1, min(i_start + n_frames, len(phi_hist))):
        idx  = np.arange(cur - WINDOW, cur + WINDOW + 1) % N
        best = np.argmax(phi_hist[i][idx])
        cur  = idx[best]
        positions.append(x[cur])
        times.append(t_anim[i])
        local_h.append(phi_hist[i][cur])

    return np.array(times), np.array(positions), np.array(local_h)

def unwrap_periodic(pos):
    p = pos.copy().astype(float)
    for i in range(1, len(p)):
        d = p[i] - p[i-1]
        if   d >  L/2: p[i:] -= L
        elif d < -L/2: p[i:] += L
    return p

heights = []
speeds  = []

for pk in peaks0:
    h0 = phi_sep[pk]
    times, pos, local_h = track_soliton(i_best, pk, n_frames=200)
    pos_uw = unwrap_periodic(pos)

    stable = np.abs(local_h - h0) / h0 < 0.4
    if stable.sum() < 5:
        stable = np.ones(len(times), dtype=bool)

    v = np.polyfit(times[stable], pos_uw[stable], 1)[0]
    heights.append(h0)
    speeds.append(v)

heights = np.array(heights)
speeds  = np.array(speeds)

def linear(h, a):
    return a * h

popt, _ = curve_fit(linear, heights, speeds, p0=[0.4])
h_fit   = np.linspace(0, heights.max() * 1.1, 300)

fig_b, ax_b = plt.subplots(figsize=(7, 5))
ax_b.scatter(heights, speeds, s=60, zorder=5, label='Medido')
ax_b.plot(h_fit, linear(h_fit, *popt), 'r-', lw=2,
          label=f'Ajuste: v = {popt[0]:.3f} · h')
ax_b.plot(h_fit, h_fit / 3, 'b--', lw=1.5,
          label='Teoría KdV: v = h/3 ≈ 0.333·h')
ax_b.set_xlabel('Altura del solitón')
ax_b.set_ylabel('Velocidad')
ax_b.set_title('Velocidad vs Altura — Solitones de Korteweg-de-Vries')
ax_b.legend()
ax_b.grid(True, alpha=0.3)
ax_b.set_xlim(left=0)
ax_b.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('3.b_speed.pdf')





grid = pde.CartesianGrid([[0,3],[0,3]], [128,128], periodic=True)

u_t = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
v_t = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
state_t = pde.FieldCollection([u_t, v_t])
alpha, beta, a, b = 0.000288, 0.0567, 0.0065, 27
eq_turing = pde.PDE({
    'u_t': 'alpha * laplace(u_t) + u_t - u_t**3 - v_t - a',
    'v_t': 'beta  * laplace(v_t) + b*(u_t - v_t)'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller_t = pde.Controller(pde.EulerSolver(eq_turing), t_range=30, tracker=None)
final_t = controller_t.run(state_t, dt=0.001)

fig, ax = plt.subplots()
u_final_t = final_t[0]
ax.imshow(u_final_t.data, cmap='pink', origin='lower')
ax.set_title('α=0.00028, β=0.05, a=0.0065, b=27\n' \
'F=u-u³-v-a, G=b(u-v)', fontsize=10)
fig.savefig('2_cerebro.png', dpi=150, bbox_inches='tight')

u_l = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
v_l = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
state_l = pde.FieldCollection([u_l, v_l])
alpha, beta, a, b = 0.0005, 0.01, 0.0065, 5
eq_leopardo = pde.PDE({
    'u_l': 'alpha * laplace(u_l) + u_l - u_l**5 - v_l - a+u_l*v_l',
    'v_l': 'beta  * laplace(v_l) + b*(u_l - v_l)'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller_l = pde.Controller(pde.EulerSolver(eq_leopardo), t_range=30, tracker=None)
final_l = controller_l.run(state_l, dt=0.01)

fig, ax = plt.subplots()
u_final_l = final_l[0]
ax.imshow(u_final_l.data, cmap='inferno_r', origin='lower')
ax.set_title('α=0.0005, β=0.01, a=0.0065, b=5\n' \
'F=u-u⁵-v-a*(uv), G=b(u-v)', fontsize=10)
fig.savefig('2_leopardo.png', dpi=150, bbox_inches='tight')

u_z = pde.ScalarField.random_normal(grid, mean=1, std=0.1)
v_z = pde.ScalarField.random_normal(grid, mean=1, std=0.1)
state_z = pde.FieldCollection([u_z, v_z])
alpha, beta, a, b = 0.000288, 0.0567, 15, 0.05
eq_zebra = pde.PDE({
    'u_z': "alpha* (d2_dx2(u_z)+0.3*d2_dy2(u_z))+u_z-u_z**3/3-v_z-b",
    'v_z': "beta * laplace(v_z) + a*(u_z - v_z)"
}, consts={'alpha': alpha, 'beta': beta, "a": a, "b": b})
controller_z = pde.Controller(pde.EulerSolver(eq_zebra), t_range=30, tracker=None)
final_z = controller_z.run(state_z, dt=0.001)

fig, ax = plt.subplots()
u_final_z = final_z[0]
ax.imshow(u_final_z.data, cmap='gist_gray', origin='lower')
ax.set_title('α=0.000288, β=0.0567, a=15, b=0.05\n' \
'F=u-(u³)/3-v-b, G=a(u-v)', fontsize=10)
fig.savefig('2_zebra.png', dpi=150, bbox_inches='tight')

u_b = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
v_b = pde.ScalarField.random_normal(grid, mean=0, std=0.1)
state_b = pde.FieldCollection([u_b, v_b])
alpha, beta, a, b = 0.002, 0.065, 2.0, 1.0
eq_brussel = pde.PDE({
    'u_b': 'alpha * laplace(u_b) + a - (b+1)*u_b + u_b**2 * v_b',
    'v_b': 'beta  * laplace(v_b) + b*u_b - u_b**2 * v_b'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller_b = pde.Controller(pde.ExplicitSolver(eq_brussel), t_range=40, tracker=None)
final_b = controller_b.run(state_b, dt=0.001)

fig, ax = plt.subplots()
u_final_b = final_b[0]
ax.imshow(u_final_b.data, cmap='Spectral', origin='lower')
ax.set_title(f'α=0.002, β=0.065, a=2, b=1\n \
F=a-(b+1)u+u²v, G=bu-u²v', fontsize=10)
fig.savefig('2_brusselator(a_veces_patas_de_rana).png', dpi=150, bbox_inches='tight')

u_gm = pde.ScalarField.random_normal(grid, mean=1, std=0.5)
v_gm = pde.ScalarField.random_normal(grid, mean=1, std=0.5)
state_gm = pde.FieldCollection([u_gm, v_gm])
alpha, beta, a, b = 0.001, 0.08, 0.5, 0.9
eq_gierer = pde.PDE({
    'u_gm': 'alpha * laplace(u_gm) + u_gm**2 / v_gm - a*u_gm',
    'v_gm': 'beta  * laplace(v_gm) + u_gm**2 - b*v_gm'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller_gm = pde.Controller(pde.ExplicitSolver(eq_gierer), t_range=30, tracker=None)
final_gm = controller_gm.run(state_gm, dt=0.001)

fig, ax = plt.subplots()
u_final_gm = final_gm[0]
ax.imshow(u_final_gm.data, cmap='bone', origin='lower')
ax.set_title(f'α=0.001, β=0.08, a=0.5, b=0.9\n \
F=u²/v-au, G=u²-bv', fontsize=10)
fig.savefig('2_piel_manta_raya(o_noche_estrellada).png', dpi=150, bbox_inches='tight')