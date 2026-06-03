import pandas as pd 
import matplotlib.pyplot as plt 
import h5py
import scipy
from scipy.optimize import curve_fit
import numpy as np
from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import quad
from scipy.optimize import root_scalar
import os

rutas = {
    "Mo": Path("./mammography_spectra/Mo_unfiltered_10kV-50kV"),
    "Rh": Path("./mammography_spectra/Rh_unfiltered_10kV-50kV"),
    "W":  Path("./mammography_spectra/W_unfiltered_10kV-50kV"),
}

#convertimos a dataframes
dfs = {}

for material, ruta in rutas.items():
    dfs[material] = {}
    #print(dfs)

    for archivo in ruta.glob("*.dat"):
        df = pd.read_csv(
            archivo,
            encoding="ISO8859",
            sep="\t",
            comment="#",
            names=["en"
            "ergy", "flux"]
        )
        
        dfs[material][archivo.stem] = df

#nos toca guardar los parametros de cada dataframe 
flux_max = {}
kVs_spline = {}
E_max = {}
curvaturas = {}
FWHM_all = {}
skew_all = {}
kurto_all = {}
amplitud = {}
sigma_p2 = {}
kVs_amp = {}
fraccion = {}

#resultados = {}
#kVs = {}
#max_spline = {}


#eso lo mandamos arbitrario porque es la ventana en la que vamos a ver los datos 
E_RANGE = {
    "Mo": (16, 20),
    "W":  (5, 20),
    "Rh": (19, 26),
}

def make_smoothing_spline_positive(*args,**kwargs):
    xd = make_smoothing_spline(*args,**kwargs)
    return np.vectorize(lambda x: xd(x) if xd(x)>0.0 else 0.0)

def curvatura(x,y):
    d1 = np.gradient(y,x)
    d2 = np.gradient(d1,x)
    return d2/((1+d1**2)**(3/2))


for material, archivos in dfs.items():
    #resultados[material] = {}
    #kVs[material] = []
    flux_max[material] = []
    #max_spline[material] = {}
    kVs_spline[material] =[]
    E_max[material] =[]
    curvaturas[material] = []
    FWHM_all[material] =[]
    skew_all[material] = []
    kurto_all[material] = []
    amplitud[material] = []
    sigma_p2[material] = []
    kVs_amp[material] = []
    fraccion[material] = []

    e_min, e_max = E_RANGE.get(material, (-np.inf, np.inf))


    for nombre, df in archivos.items():
        x = df["energy"].to_numpy()
        y = df["flux"].to_numpy()

        
        peaks, props = find_peaks(y, prominence=0.05)


        #filtramos pico por energia x[peaks]
        peaks_filtrados = peaks[(x[peaks] >= e_min) & (x[peaks] <= e_max)]
        #print(peaks_filtrados)
     
        mask = np.ones(len(y),dtype=np.bool)
        
        if len(peaks_filtrados) > 0:
            for i in peaks_filtrados: 

                mask[i] = 0 
                mask[i+1] = 0
                mask[i-1] =0
                mask[i+2]=0
                mask[i-2]=0
                mask[i+3] =0
                mask[i-3]=0
                mask[i+4] =0
                mask[i-4]=0

        x_bg = x[mask]
        y_bg = y[mask]

        spline_bg = make_smoothing_spline_positive(x_bg, y_bg, lam=0.5)

      
        # 1) & 2) SACAR EL MAXIMO DE FLUX y ENERGIA
        kv = int(nombre.split("_")[1].replace("kV", ""))
        en = np.linspace(x.min(), x.max(), 5000)
        y_spline = spline_bg(en)

        idx_max = np.argmax(y_spline)
        f_max = y_spline.max()
        e_max_flux = en[idx_max]

        kVs_spline[material].append(kv)
        flux_max[material].append(f_max)
        E_max[material].append(e_max_flux)

        #3)CURVATURA
        k = curvatura(en, y_spline)
        k_max = k[idx_max]
        curvaturas[material].append(k_max)
        
        #4) ANCHO MEDIA ALTURA
        widths, _, left_ips, right_ips = peak_widths(y_spline, [idx_max])
        E1 = np.interp(right_ips[0], np.arange(len(en)),en)
        E2 = np.interp(left_ips[0], np.arange(len(en)),en)
        FWHM = E1 - E2
        FWHM_all[material].append(FWHM)


        #5) ASIMETRIA (Skewness) y KURTOSIS
        a, b = float(x.min()), float(x.max()) 
        def F(E):
            return float(spline_bg(E))

        B, _ = quad(F, a, b, limit=200)

        if B <= 0 or not np.isfinite(B):
            skew = np.nan
            kurt = np.nan
        else:
            E_hat, _ = quad(lambda E: (F(E)/B) * E, a, b, limit=200)

            var, _ = quad(lambda E: (F(E)/B) * (E - E_hat)**2, a, b, limit=200)
            sigma = np.sqrt(var)

            if sigma == 0 or not np.isfinite(sigma):
                skew = np.nan
                kurt = np.nan
            else:
                skew, _ = quad(lambda E: (F(E)/B) * ((E - E_hat)/sigma)**3, a, b, limit=200)
                kurt, _ = quad(lambda E: (F(E)/B) * ((E - E_hat)/sigma)**4, a, b, limit=200)

        skew_all[material].append(skew)
        kurto_all[material].append(kurt)

        #6 & 7 - AMPLITUD Y DESVIACIÓN ESTANDAR
        solo_barriga = spline_bg(x)
        datos_original = y

        solo_picos = datos_original - solo_barriga

        def gaussiana(x, amp, mu, sigma_2): 
            return amp * np.exp(-(x - mu)**2 / (2 * sigma_2**2))
        
        def varias_gaussianas(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
            gauss1 = gaussiana(x, A1, mu1, sigma1)
            gauss2 = gaussiana(x, A2, mu2, sigma2)
            gauss3 = gaussiana(x, A3, mu3, sigma3)
            return gauss1 + gauss2 + gauss3
        
        if len(peaks_filtrados) > 0:
            #ordenar de maayor a menor, mejor porque piede ser que lo lee de izq a der
            indices_ordenados = np.argsort(solo_picos[peaks_filtrados])[::-1]
            peaks_ordenados = peaks_filtrados[indices_ordenados]
            
            mu_inicial = x[peaks_ordenados]
            amplitud_inicial = solo_picos[peaks_ordenados]

            rango = e_max-e_min

            p0 = None

            if material == "W":
                if len(peaks_filtrados) >= 3:
                    sigma_inicial = rango/3
                    p0 = [amplitud_inicial[0], mu_inicial[0], sigma_inicial,
                        amplitud_inicial[1], mu_inicial[1], sigma_inicial,
                        amplitud_inicial[2], mu_inicial[2], sigma_inicial]
            else:
                if len(peaks_filtrados) >= 2:
                    sigma_inicial = rango/2
                    p0 = [amplitud_inicial[0], mu_inicial[0], sigma_inicial,
                        amplitud_inicial[1], mu_inicial[1], sigma_inicial,
                        0, 0, 0]                
            
            if p0 is not None:
                x_picos = x[(x >= e_min) & (x <= e_max)]
                y_picos = solo_picos[(x >= e_min) & (x <= e_max)]
            
                try:
                    parametros, _ = curve_fit(
                        varias_gaussianas, 
                        x_picos, 
                        y_picos, 
                        p0=p0,
                        bounds=(
                            [0, -np.inf, 0, 0, -np.inf, 0, 0, -np.inf, 0],
                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                        maxfev=5000
                    )

                    A1, mu1, sigma1 = parametros[0], parametros[1], parametros[2]
                    A2, mu2, sigma2 = parametros[3], parametros[4], parametros[5]
                    A3, mu3, sigma3 = parametros[6], parametros[7], parametros[8] 

                    amplitud[material].append([A1, A2, A3])
                    sigma_p2[material].append([sigma1, sigma2, sigma3])
                    kVs_amp[material].append(kv)

                except RuntimeError:
                    pass   
        
        # 8) FRACCIÓN DE ÁREA
        try:
            def picos_ajustados(E):
                return varias_gaussianas(E, *parametros)
            
            area_picos, _ = quad(picos_ajustados, a, b, limit=200)
            frac = area_picos / B

            fraccion[material].append(frac)
            
        except NameError:
            pass



# helper para ordenar por kV
def sort_xy(kv_list, y_list):
    kv = np.array(kv_list)
    yy = np.array(y_list)
    idx = np.argsort(kv)
    return kv[idx], yy[idx]

# figura con subplots
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
axs = axs.flatten()

# 1) Flux máximo vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], flux_max[material])
    axs[0].plot(kv, yy, "o-", label=material)
axs[0].set_title("Flux máximo vs kV")
axs[0].set_xlabel("kV")
axs[0].set_ylabel("Flux máximo")
axs[0].legend()

# 2) E_max vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], E_max[material])
    axs[1].plot(kv, yy, "o-", label=material)
axs[1].set_title("E_max vs kV")
axs[1].set_xlabel("kV")
axs[1].set_ylabel("E_max [keV]")
axs[1].legend()

# 3) Curvatura vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], curvaturas[material])
    axs[2].plot(kv, yy, "o-", label=material)
axs[2].set_title("Curvatura vs kV")
axs[2].set_xlabel("kV")
axs[2].set_ylabel(r"$\kappa$")
#axs[2].legend()

# 4) FWHM vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], FWHM_all[material])
    axs[3].plot(kv, yy, "o-", label=material)
axs[3].set_title("FWHM vs kV")
axs[3].set_xlabel("kV")
axs[3].set_ylabel("FWHM [keV]")
#axs[3].legend()

# 5) Skewness vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], skew_all[material])
    axs[4].plot(kv, yy, "o-", label=material)
axs[4].set_title("Skewness vs kV")
axs[4].set_xlabel("kV")
axs[4].set_ylabel("Skewness")
#axs[4].legend()

#6) Kurtosis vs kV
for material in ["Mo", "Rh", "W"]:
    kv, yy = sort_xy(kVs_spline[material], kurto_all[material])
    axs[5].plot(kv, yy, "o-", label=material)
axs[5].set_title("Kurtosis vs kV")
axs[5].set_xlabel("kV")
axs[5].set_ylabel("Kurtosis")
#axs[5].legend()

# 7) Amplitud vs kV
for material in ["Mo", "Rh", "W"]:
    if material in amplitud and len(amplitud[material]) > 0:
        kv, amps = sort_xy(kVs_amp[material], amplitud[material])
        amps = np.array(amps)
        axs[6].plot(kv, amps[:, 0], 'o-', label=f'{material} Pico 1')
        axs[6].plot(kv, amps[:, 1], 's-', label=f'{material} Pico 2')
        if material == "W":
            axs[6].plot(kv, amps[:, 2], '^-', label=f'{material} Pico 3')
axs[6].set_title("Amplitud vs kV")
axs[6].set_xlabel("kV")
axs[6].set_ylabel("Amplitud")
axs[6].legend()

# 8) Sigma (desviación estándar) vs kV
for material in ["Mo", "Rh", "W"]:
    if material in sigma_p2 and len(sigma_p2[material]) > 0:
        kv, sigs = sort_xy(kVs_amp[material], sigma_p2[material])
        sigs = np.array(sigs)
        axs[7].plot(kv, sigs[:, 0], 'o-', label=f'{material} Pico 1')
        axs[7].plot(kv, sigs[:, 1], 's-', label=f'{material} Pico 2')
        if material == 'W':
            axs[7].plot(kv, sigs[:, 2], '^-', label=f'{material} Pico 3')
axs[7].set_title("Sigma vs kV")
axs[7].set_xlabel("kV")
axs[7].set_ylabel("Sigma")
#axs[7].legend()

# 9) Fracción de área vs kV
for material in ["Mo", "Rh", "W"]:
    if material in fraccion and len(fraccion[material]) > 0:
        kv, fracs = sort_xy(kVs_amp[material], fraccion[material])
        axs[8].plot(kv, fracs, 'o-', label=f'{material}')

axs[8].set_title("Fracción de Área de Picos vs kV")
axs[8].set_xlabel("kV")
axs[8].set_ylabel("Fracción de Área")
#axs[8].legend()

# 10) Gráfica de Skew vs Kurtosis
for material in ["Mo", "Rh", "W"]:
    if material in skew_all and material in kurto_all:
        skews = np.array(skew_all[material])
        kurts = np.array(kurto_all[material])
        
        mask_valido = ~(np.isnan(skews) | np.isnan(kurts))
        
        if np.sum(mask_valido) > 0:
            axs[9].plot(skews[mask_valido], kurts[mask_valido], 'o-', label=f'{material}')

axs[9].set_title("Skewness vs Kurtosis")
axs[9].set_xlabel("Skewness")
axs[9].set_ylabel("Kurtosis")
#axs[9].legend()

#cositas estetik
axs[10].axis('off')
axs[11].axis('off') 

for ax in axs:
    ax.title.set_fontsize(12)      
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.tick_params(labelsize=9)
    if ax.get_legend():
        ax.legend(fontsize=8)

plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)


#plt.show()

#PDF
plt.tight_layout()

plt.savefig("rayos_X.pdf", format='pdf', bbox_inches='tight', dpi=300)
