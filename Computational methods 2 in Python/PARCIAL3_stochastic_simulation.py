import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numba import njit

random = np.random.default_rng()





def simular_puntaje(p, vidas=3): 
    caras =0
    sellos = 0
    while caras<=vidas: 
        moneda= np.random.random()
        
        if moneda < p: 
            caras+=1
        else: 
            sellos+=1
    
    return sellos 

p_trampa = 0.18
vidas = 3
N= 10_000


resultados = [simular_puntaje(p_trampa, vidas) for i in range(N)]
ks = np.arange(0, 80)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(resultados, bins=ks, density=True)
plt.plot(ks, stats.nbinom(4, p_trampa).pmf(ks), c='k')
ax.set_xlabel('Número de cofres abiertos')  
ax.set_ylabel('Densidad de probabilidad')  
plt.savefig("1.pdf")
plt.show()












lam = 80
mu = 2

@njit
def SSA_step(state, reactions, rates):
    total_rate = rates.sum()
    normalized = rates / total_rate
    u = np.random.rand()
    k = np.searchsorted(np.cumsum(normalized), u)
    τ = np.random.exponential(scale=1/total_rate)
    new_state = state + reactions[k]
    return τ, new_state

@njit
def rate_function(state):
    n = state[0]
    return np.array([lam, n * mu])

@njit
def system_simulation(state, reactions, rate_function, iters, t_max):
    states = np.empty((iters, len(state)), dtype=state.dtype)
    states[0] = state
    times = np.zeros(iters)
    rates = rate_function(state)
    for i in range(1, iters):
        τ, states[i] = SSA_step(states[i-1], reactions, rates)
        times[i] = times[i-1] + τ
        rates = rate_function(states[i])
        if times[i] >= t_max:
            return times[:i+1], states[:i+1]
    return times, states

reactions = np.array([[1], [-1]])

fig, ax = plt.subplots(figsize=(12, 6))
valores_estable = []

for j in range(5):
    times, results = system_simulation(np.array([40]), reactions, rate_function, 50000, t_max=10.0)
    ax.plot(times, results[:, 0], lw=0.5, label=f'Simulación {j+1}')

    mask = times >= 5
    valores_estable.extend(results[mask, 0])

std_estable = np.std(valores_estable)
promedio_estable = np.mean(valores_estable)

ax.axhline(lam/mu, c='k', ls='--', linewidth=1.5, label=f'Promedio teórico = {lam/mu:.0f}')
ax.set_xlabel('tiiempo (s)')
ax.set_ylabel('Numero de enemigos')
ax.set_title(f'λ={lam}/s,' f'std={std_estable:.2f}')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('2.a.pdf')
plt.show()




def f(x, mu):
    if x > 0:
        return np.exp(-x / mu)
    else:
        return 0.0

def expon(promedio=1.0, n=100):
    mu = promedio
    N = 100_000

    muestras = np.zeros(N)
    f_values = np.zeros(N)

    muestras[-1] = mu       
    f_values[-1] = f(muestras[-1], mu)
    σ_perturb = mu * 1.5

    n_accepted = 0

    for i in range(N):
        old = muestras[i-1]
        new = old + random.normal(loc=0.0, scale=σ_perturb)
        f_old = f_values[i-1]
        f_new = f(new, mu)

        if f_old != 0 and random.random() < f_new / f_old:
            muestras[i] = new
            f_values[i] = f_new
            n_accepted += 1
        else:
            muestras[i] = old
            f_values[i] = f_old

    return muestras[1000::10][:n]


muestras = expon(promedio=2.0, n=100)
print(muestras)












def simular_puntaje(vidas=3):
    puntos = 0
    
    while vidas > 0:
        cofre = random.random()

        if cofre < 0.18:
            vidas -= 1     

        elif cofre < 0.18 + 0.07:
            vidas += 1     
        elif cofre < 0.18 + 0.07 + 0.60:
            puntos += 1      

        else:
            puntos += 2     


        habitacion = random.random()
        if habitacion < 0.40:
            vidas -= 0
        elif habitacion < 0.40 + 0.30:
            vidas -= 1
        elif habitacion < 0.40 + 0.30 + 0.20:
            vidas -= 2
        else:
            vidas -= 3

    return puntos


resultados = [simular_puntaje(vidas=3) for i in range(10_000)]
print(np.mean(resultados))
print(np.max(resultados))
print(np.min(resultados))


