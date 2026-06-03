import pde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

grid = pde.CartesianGrid([[0,3],[0,3]], [128,128], periodic=True)

n_frames = 200
storage  = pde.MemoryStorage()
t_final = 35
rng = np.random.default_rng(42)

u_t = pde.ScalarField.random_normal(grid, rng.normal(0, 0.1, (128, 128)))
v_t = pde.ScalarField.random_normal(grid, rng.normal(0, 0.1, (128, 128)))
state_t = pde.FieldCollection([u_t, v_t])
alpha, beta, a, b = 0.000288, 0.0567, 0.0065, 27
eq_turing = pde.PDE({
    'u_t': 'alpha * laplace(u_t) + u_t - u_t**3 - v_t - a',
    'v_t': 'beta  * laplace(v_t) + b*(u_t - v_t)'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller = pde.Controller(pde.ExplicitSolver(eq_turing), t_range=t_final, tracker=storage.tracker(t_final / n_frames))
final_t = controller.run(state_t, dt=0.001)

fig, ax = plt.subplots(figsize=(5, 5))
img = ax.imshow(
    storage[0][1].data,
    origin='lower',
    extent=[0, 3, 0, 3],
    cmap='pink',
    vmin=-1, vmax=1,
    animated=True,
)
ax.set_axis_off()
title = ax.set_title('t = 0.00')
fig.text(0.02, 0.02,
    'α=0.00028, β=0.05, a=0.0065, b=27\nF=u-u³-v-a, G=b(u-v)',
    fontsize=7, verticalalignment='bottom', fontfamily='monospace')

def update_cerebro(frame):
    img.set_data(storage[frame][0].data)
    title.set_text(f't = {frame * t_final/n_frames:.2f}')
    return img, title

n_frames = len(storage)
anim = animation.FuncAnimation(fig, update_cerebro, frames=n_frames, interval=40, blit=True)
writer = animation.FFMpegWriter(fps=20, bitrate=1800)
anim.save('2_cerebro_animacion.mp4', writer=writer, dpi=150)



u_l = pde.ScalarField.random_normal(grid, rng.normal(0, 0.1, (128, 128)))
v_l = pde.ScalarField.random_normal(grid, rng.normal(0, 0.1, (128, 128)))
state_l = pde.FieldCollection([u_l, v_l])
alpha, beta, a, b = 0.0005, 0.01, 0.0065, 5
eq_leopardo = pde.PDE({
    'u_l': 'alpha * laplace(u_l) + u_l - u_l**5 - v_l - a+u_l*v_l',
    'v_l': 'beta  * laplace(v_l) + b*(u_l - v_l)'
}, consts={'alpha': alpha, 'beta': beta, 'a': a, 'b': b})
controller_l = pde.Controller(pde.EulerSolver(eq_leopardo), t_range=t_final, tracker=storage.tracker(t_final / n_frames))
final_l = controller_l.run(state_l, dt=0.01)

fig, ax = plt.subplots(figsize=(5, 5))
img = ax.imshow(
    storage[0][1].data,
    origin='lower',
    extent=[0, 3, 0, 3],
    cmap='inferno_r',
    vmin=-1, vmax=1,
    animated=True,
)
ax.set_axis_off()
title = ax.set_title('t = 0.00')
fig.text(0.02, 0.02,
    'α=0.0005, β=0.01, a=0.0065, b=5\n''F=u-u⁵-v-a*(uv), G=b(u-v)',
    fontsize=7, verticalalignment='bottom', fontfamily='monospace')

def update_leopardo(frame):
    img.set_data(storage[frame][0].data)
    title.set_text(f't = {frame * t_final/n_frames:.2f}')
    return img, title

n_frames = len(storage)
anim = animation.FuncAnimation(fig, update_leopardo, frames=n_frames, interval=40, blit=True)
writer = animation.FFMpegWriter(fps=20, bitrate=1800)
anim.save('2_leopardo_animacion.mp4', writer=writer, dpi=150)