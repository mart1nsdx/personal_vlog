import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import find_peaks
import pandas as pd
from scipy import ndimage as ndi
from scipy.special import expit

x_values = np.linspace(-5,5,2000)
a_values = np.linspace(0.1,2,2000)
sigma_values = np.linspace(0.2, 2.0, 2000)
k_values = np.linspace(-2,2,500)


def box_a(x, a):
    return np.abs(x)<a/2

def Fourier_transform(x, y, k):
    return np.sum(y*np.exp(-2j*np.pi*x*k))

def gaussian(x,sigma): 
    return np.exp(-0.5*(x/sigma)**2)



fwhm_k_box_a = np.zeros_like(a_values)
fwhm_x_box = a_values.copy()

for i,a in enumerate(a_values):
    y = box_a(x_values, a)
    def ceros_de_la_caja(k):
        return np.abs(Fourier_transform(x_values,y,k)) - np.abs(Fourier_transform(x_values,y,0))/2 

    raices = root_scalar(ceros_de_la_caja, bracket=[0.1,20]).root
    FWHM_transformada_caja = raices*2

    fwhm_k_box_a[i] = (FWHM_transformada_caja)


fwhm_x_gauss = np.array([2*s*np.sqrt(2*np.log(2)) for s in sigma_values])
fwhm_k_gauss = np.zeros_like(sigma_values)

for i, s in enumerate(sigma_values):
    y = gaussian(x_values, s)
    F0 = np.abs(Fourier_transform(x_values,y,0))
    half= F0/2
    def ceros_gaussiana(k): 
        return np.abs(Fourier_transform(x_values,y,k))-half
    
    roots = root_scalar(ceros_gaussiana, bracket=[1e-6, k_values.max()]).root
    fwhm_k_gauss[i] = 2*roots


def ajuste(x,m,b): 
    return np.exp(b)*(x**m)

param_box,_ = curve_fit(ajuste, a_values, fwhm_k_box_a)
param_gauss,_ = curve_fit(ajuste, sigma_values, fwhm_k_gauss)


plt.figure(figsize=(6,4))

plt.scatter(a_values, fwhm_k_box_a, s=10)
plt.plot(a_values, ajuste(fwhm_k_box_a, *param_box), lw=2)

plt.xlabel("a")
plt.ylabel("FWHM_k")
plt.title("Box")

plt.tight_layout()
plt.savefig("1-box.pdf")
plt.close()


plt.figure(figsize=(6,4))

plt.scatter(sigma_values, fwhm_k_gauss, s=10)
plt.plot(sigma_values, ajuste(fwhm_k_gauss, *param_gauss), lw=2)

plt.xlabel("sigma")
plt.ylabel("FWHM_k")
plt.title("Gauss")

plt.tight_layout()
plt.savefig("1-gauss.pdf")
plt.close()








data = np.load("star.npy")

df_dias = data[0]
df_brillo = data[1]

T = df_dias.max() - df_dias.min()
df = 0.5 / T 
freq_values = np.arange(0, 4.0, df)

brillo_0 = df_brillo - np.mean(df_brillo)

def fit_fourier_series(t, y, f, K=5):

    t = np.asarray(t)
    y = np.asarray(y)

    cols = [np.ones_like(t)]
    w = 2*np.pi*f*t
    for k in range(1, K+1):
        cols.append(np.cos(k*w))
        cols.append(np.sin(k*w))
    A = np.column_stack(cols)


    coeffs,*_ = np.linalg.lstsq(A, y)

    def yhat(t_new):
        t_new = np.asarray(t_new)
        cols_new = [np.ones_like(t_new)]
        w_new = 2*np.pi*f*t_new
        for k in range(1, K+1):
            cols_new.append(np.cos(k*w_new))
            cols_new.append(np.sin(k*w_new))
        A_new = np.column_stack(cols_new)
        return A_new @ coeffs

    return coeffs, yhat

espectro_datos = np.array([Fourier_transform(df_dias, brillo_0, f) for f in freq_values])
amp0 = np.abs(espectro_datos)
amp0[0] = 0
mask_1day = np.abs(freq_values - 1.0) < 5*df
amp0[mask_1day] = 0
peaks0, _ = find_peaks(amp0, prominence=np.max(amp0)*0.05, distance=5)
f0 = freq_values[peaks0[np.argmax(amp0[peaks0])]]

K0 = 5
coeff0, f0_model = fit_fourier_series(df_dias, brillo_0, f0, K=K0)
fit0 = f0_model(df_dias)
residuo = brillo_0 - fit0



espectro_res = np.array([Fourier_transform(df_dias, residuo, f) for f in freq_values])
amp1 = np.abs(espectro_res)
amp1[0] = 0
amp1[mask_1day] = 0
mask_f0 = np.abs(freq_values - f0) < 5*df
amp1[mask_f0] = 0
peaks1, _ = find_peaks(amp1, prominence=np.max(amp1)*0.05, distance=5)
f1 = freq_values[peaks1[np.argmax(amp1[peaks1])]]

K1 = 5
coeff1, f1_model = fit_fourier_series(df_dias, residuo, f1, K=K1)
fit1 = f1_model(df_dias)


modelo_completo = fit0 + fit1


def phase_diagram(t, f):
    return (t*f) % 1.0

phi0 = phase_diagram(df_dias, f0)
y0_isolated = brillo_0 - fit1
order0 = np.argsort(phi0)

phi1 = phase_diagram(df_dias, f1)
y1_isolated = brillo_0 - fit0
order1 = np.argsort(phi1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
ax[0].scatter(phi0, y0_isolated, s=8)
ax[0].plot(phi0[order0], f0_model(df_dias)[order0],color="r")
ax[0].set_xlabel("phase f0")
ax[0].set_ylabel("brillo")
ax[1].scatter(phi1, y1_isolated, s=8)
ax[1].plot(phi1[order1], f1_model(df_dias)[order1],color= "r")
ax[1].set_xlabel("phase f1")
ax[1].set_ylabel("brillo")
plt.savefig("2.A.harmonics.png")
plt.close()


cargar_carpeta = np.load("images.npz")
[cargar_carpeta.keys()]


imagen_1 = cargar_carpeta["Etimologizante"]

imagen_1_b = cargar_carpeta["Etimologizante"][:,:,2]
imagen_1_b_transformada = np.fft.fft2(imagen_1_b)
imagen_1_b_transformada_shift = np.fft.fftshift(imagen_1_b_transformada)
imagen_1_b_transformada_shift[127,75] = 0.0
imagen_1_b_transformada_shift[1,53] = 0.0
imagen_1_b_transformada_shift[23,48] = 0.0
imagen_1_b_transformada_shift[49,43] = 0.0
imagen_1_b_transformada_shift[79,85] = 0.0
imagen_1_b_transformada_shift[105,80] = 0.0
imagen_1_b_transformada_y_filtrada = imagen_1_b_transformada_shift
imagen_1_b_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_1_b_transformada_y_filtrada))

imagen_1_g = cargar_carpeta["Etimologizante"][:,:,1]
imagen_1_g_transformada = np.fft.fft2(imagen_1_g)
imagen_1_g_transformada_shift = np.fft.fftshift(imagen_1_g_transformada)
imagen_1_g_transformada_shift[54,32] = 0.0
imagen_1_g_transformada_shift[52,50] = 0.0
imagen_1_g_transformada_shift[74,96] = 0.0
imagen_1_g_transformada_shift[76,78] = 0.0
imagen_1_g_transformada_y_filtrada = imagen_1_g_transformada_shift
imagen_1_g_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_1_g_transformada_y_filtrada))

imagen_1_r = cargar_carpeta["Etimologizante"][:,:,0]
imagen_1_r_transformada = np.fft.fft2(imagen_1_r)
imagen_1_r_transformada_shift = np.fft.fftshift(imagen_1_r_transformada)
para_final_1_r =imagen_1_r_transformada_shift
imagen_1_r_transformada_shift[36,5] = 0.0
imagen_1_r_transformada_shift[48,6] = 0.0
imagen_1_r_transformada_shift[58,53] = 0.0
imagen_1_r_transformada_shift[70,75] = 0.0
imagen_1_r_transformada_shift[80,122] = 0.0
imagen_1_r_transformada_shift[92,123] = 0.0
imagen_1_r_transformada_y_filtrada = imagen_1_r_transformada_shift
imagen_1_r_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_1_r_transformada_y_filtrada))

imagen_final = np.stack([imagen_1_r_filtrada_en_2d, imagen_1_g_filtrada_en_2d, imagen_1_b_filtrada_en_2d], axis=2)
imagen_final = imagen_final.real
imagen_final = (imagen_final - imagen_final.min()) / (imagen_final.max() - imagen_final.min())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(imagen_1)
ax[0].axis('off')
ax[1].imshow(np.log(np.abs(para_final_1_r)))
ax[1].axis('off')
ax[2].imshow(np.log(np.abs(imagen_1_r_transformada_y_filtrada)))
ax[2].axis('off')
ax[3].imshow(imagen_final)
ax[3].axis('off')
plt.tight_layout()
plt.savefig('3.A.1.png', dpi=150, bbox_inches='tight')
plt.close()


imagen_2 = cargar_carpeta["Paraba"]

imagen_2_b = cargar_carpeta["Paraba"][:,:,2]
imagen_2_b_transformada = np.fft.fft2(imagen_2_b)
imagen_2_b_transformada_shift = np.fft.fftshift(imagen_2_b_transformada)
imagen_2_b_transformada_shift[7,54] = 0.0
imagen_2_b_transformada_shift[31,52] = 0.0
imagen_2_b_transformada_shift[31,19] = 0.0
imagen_2_b_transformada_shift[43,58] = 0.0
imagen_2_b_transformada_shift[85,70] = 0.0
imagen_2_b_transformada_shift[97,76] = 0.0
imagen_2_b_transformada_shift[97,109] = 0.0
imagen_2_b_transformada_shift[121,74] = 0.0
imagen_2_b_transformada_y_filtrada = imagen_2_b_transformada_shift
imagen_2_b_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_2_b_transformada_y_filtrada))

imagen_2_g = cargar_carpeta["Paraba"][:,:,1]
imagen_2_g_transformada = np.fft.fft2(imagen_2_g)
imagen_2_g_transformada_shift = np.fft.fftshift(imagen_2_g_transformada)
imagen_2_g_transformada_shift[35,50] = 0.0
imagen_2_g_transformada_shift[93,78] = 0.0
imagen_2_g_transformada_y_filtrada = imagen_2_g_transformada_shift
imagen_2_g_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_2_g_transformada_y_filtrada))

imagen_2_r = cargar_carpeta["Paraba"][:,:,0]
imagen_2_r_transformada = np.fft.fft2(imagen_2_r)
imagen_2_r_transformada_shift = np.fft.fftshift(imagen_2_r_transformada)
imagen_2_r_transformada_shift[54,34] = 0.0
imagen_2_r_transformada_shift[74,94] = 0.0
imagen_2_r_transformada_y_filtrada = imagen_2_r_transformada_shift
imagen_2_r_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_2_r_transformada_y_filtrada))
imagen_final_2 = np.stack([imagen_2_r_filtrada_en_2d, imagen_2_g_filtrada_en_2d, imagen_2_b_filtrada_en_2d], axis=2)
imagen_final_2 = imagen_final_2.real
imagen_final_2 = (imagen_final_2 - imagen_final_2.min()) / (imagen_final_2.max() - imagen_final_2.min())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(imagen_2)
ax[0].axis('off')
ax[1].imshow(np.log(np.abs(imagen_2_b_transformada_shift)))
ax[1].axis('off')
ax[2].imshow(np.log(np.abs(imagen_2_b_transformada_y_filtrada)))
ax[2].axis('off')
ax[3].imshow(imagen_final_2)
ax[3].axis('off')
plt.tight_layout()
plt.savefig('3.A.2.png', dpi=150, bbox_inches='tight')
plt.close()


imagen_3 = cargar_carpeta["Chichito"]

imagen_3_b = cargar_carpeta["Chichito"][:,:,2]
imagen_3_b_transformada = np.fft.fft2(imagen_3_b)
imagen_3_b_transformada_shift = np.fft.fftshift(imagen_3_b_transformada)
imagen_3_b_transformada_shift[43,59] = 0.0
imagen_3_b_transformada_shift[85,69] = 0.0
imagen_3_b_transformada_y_filtrada = imagen_3_b_transformada_shift
imagen_3_b_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_3_b_transformada_y_filtrada))

imagen_3_g = cargar_carpeta["Chichito"][:,:,1]
imagen_3_g_transformada = np.fft.fft2(imagen_3_g)
imagen_3_g_transformada_shift = np.fft.fftshift(imagen_3_g_transformada)
imagen_3_g_transformada_shift[26,33] = 0.0
imagen_3_g_transformada_shift[41,5] = 0.0
imagen_3_g_transformada_shift[44,2] = 0.0
imagen_3_g_transformada_shift[84,126] = 0.0
imagen_3_g_transformada_shift[87,123] = 0.0
imagen_3_g_transformada_shift[102,95] = 0.0
imagen_3_g_transformada_y_filtrada = imagen_3_g_transformada_shift
imagen_3_g_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_3_g_transformada_y_filtrada))

imagen_3_r = cargar_carpeta["Chichito"][:,:,0]
imagen_3_r_transformada = np.fft.fft2(imagen_3_r)
imagen_3_r_transformada_shift = np.fft.fftshift(imagen_3_r_transformada)
imagen_3_r_transformada_shift[1,0] = 0.0
imagen_3_r_transformada_shift[16,52] = 0.0
imagen_3_r_transformada_shift[112,76] = 0.0
imagen_3_r_transformada_shift[127,0] = 0.0
imagen_3_r_transformada_y_filtrada = imagen_3_r_transformada_shift
imagen_3_r_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_3_r_transformada_y_filtrada))

imagen_final_3 = np.stack([imagen_3_r_filtrada_en_2d, imagen_3_g_filtrada_en_2d, imagen_3_b_filtrada_en_2d], axis=2)
imagen_final_3 = imagen_final_3.real
imagen_final_3 = (imagen_final_3 - imagen_final_3.min()) / (imagen_final_3.max() - imagen_final_3.min())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(imagen_3)
ax[0].axis('off')
ax[1].imshow(np.log(np.abs(imagen_3_g_transformada_shift)))
ax[1].axis('off')
ax[2].imshow(np.log(np.abs(imagen_3_g_transformada_y_filtrada)))
ax[2].axis('off')
ax[3].imshow(imagen_final_3)
ax[3].axis('off')
plt.tight_layout()
plt.savefig('3.A.3.png', dpi=150, bbox_inches='tight')
plt.close()


imagen_4 = cargar_carpeta["Jubo"]

imagen_4_b = cargar_carpeta["Jubo"][:,:,2]
imagen_4_b_transformada = np.fft.fft2(imagen_4_b)
imagen_4_b_transformada_shift = np.fft.fftshift(imagen_4_b_transformada)
imagen_4_b_transformada_shift[16,59] = 0.0
imagen_4_b_transformada_shift[112,69] = 0.0
imagen_4_b_transformada_y_filtrada = imagen_4_b_transformada_shift
imagen_4_b_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_4_b_transformada_y_filtrada))

imagen_4_g = cargar_carpeta["Jubo"][:,:,1]
imagen_4_g_transformada = np.fft.fft2(imagen_4_g)
imagen_4_g_transformada_shift = np.fft.fftshift(imagen_4_g_transformada)
imagen_4_g_transformada_shift[10,1] = 0.0
imagen_4_g_transformada_shift[118,127] = 0.0
imagen_4_g_transformada_y_filtrada = imagen_4_g_transformada_shift
imagen_4_g_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_4_g_transformada_y_filtrada))

imagen_4_r = cargar_carpeta["Jubo"][:,:,0]
imagen_4_r_transformada = np.fft.fft2(imagen_4_r)
imagen_4_r_transformada_shift = np.fft.fftshift(imagen_4_r_transformada)
imagen_4_r_transformada_y_filtrada = imagen_4_r_transformada_shift
imagen_4_r_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_4_r_transformada_y_filtrada))

imagen_final_4 = np.stack([imagen_4_r_filtrada_en_2d, imagen_4_g_filtrada_en_2d, imagen_4_b_filtrada_en_2d], axis=2)
imagen_final_4 = imagen_final_4.real
imagen_final_4 = (imagen_final_4 - imagen_final_4.min()) / (imagen_final_4.max() - imagen_final_4.min())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(imagen_4)
ax[0].axis('off')
ax[1].imshow(np.log(np.abs(imagen_4_b_transformada_shift)))
ax[1].axis('off')
ax[2].imshow(np.log(np.abs(imagen_4_b_transformada_y_filtrada)))
ax[2].axis('off')
ax[3].imshow(imagen_final_4)
ax[3].axis('off')
plt.tight_layout()
plt.savefig('3.A.4.png', dpi=150, bbox_inches='tight')
plt.close()


imagen_5 = cargar_carpeta["Guiguí"]

imagen_5_b = cargar_carpeta["Guiguí"][:,:,2]
imagen_5_b_transformada = np.fft.fft2(imagen_5_b)
imagen_5_b_transformada_shift = np.fft.fftshift(imagen_5_b_transformada)
imagen_5_b_transformada_shift[10,38] = 0.0
imagen_5_b_transformada_shift[21,19] = 0.0
imagen_5_b_transformada_shift[107,109] = 0.0
imagen_5_b_transformada_shift[118,90] = 0.0
imagen_5_b_transformada_y_filtrada = imagen_5_b_transformada_shift
imagen_5_b_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_5_b_transformada_y_filtrada))

imagen_5_g = cargar_carpeta["Guiguí"][:,:,1]
imagen_5_g_transformada = np.fft.fft2(imagen_5_g)
imagen_5_g_transformada_shift = np.fft.fftshift(imagen_5_g_transformada)
imagen_5_g_transformada_shift[35,58] = 0.0
imagen_5_g_transformada_shift[51,53] = 0.0
imagen_5_g_transformada_shift[77,75] = 0.0
imagen_5_g_transformada_shift[93,70] = 0.0
imagen_5_g_transformada_y_filtrada = imagen_5_g_transformada_shift
imagen_5_g_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_5_g_transformada_y_filtrada))

imagen_5_r = cargar_carpeta["Guiguí"][:,:,0]
imagen_5_r_transformada = np.fft.fft2(imagen_5_r)
imagen_5_r_transformada_shift = np.fft.fftshift(imagen_5_r_transformada)
imagen_5_r_transformada_shift[3,36] = 0.0
imagen_5_r_transformada_shift[24,42] = 0.0
imagen_5_r_transformada_shift[33,1] = 0.0
imagen_5_r_transformada_shift[95,127] = 0.0
imagen_5_r_transformada_shift[104,86] = 0.0
imagen_5_r_transformada_shift[125,92] = 0.0
imagen_5_r_transformada_y_filtrada = imagen_5_r_transformada_shift
imagen_5_r_filtrada_en_2d = np.fft.ifft2(np.fft.fftshift(imagen_5_r_transformada_y_filtrada))

imagen_final_5 = np.stack([imagen_5_r_filtrada_en_2d, imagen_5_g_filtrada_en_2d, imagen_5_b_filtrada_en_2d], axis=2)
imagen_final_5 = imagen_final_5.real
imagen_final_5 = (imagen_final_5 - imagen_final_5.min()) / (imagen_final_5.max() - imagen_final_5.min())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(imagen_5)
ax[0].axis('off')
ax[1].imshow(np.log(np.abs(imagen_5_r_transformada_shift)))
ax[1].axis('off')
ax[2].imshow(np.log(np.abs(imagen_5_r_transformada_y_filtrada)))
ax[2].axis('off')
ax[3].imshow(imagen_final_5)
ax[3].axis('off')
plt.tight_layout()
plt.savefig('3.A.5.png', dpi=150, bbox_inches='tight')
plt.close()




def pasaaltas(size, k0=0.1, steepness=10):
    freqs = np.fft.fftfreq(size)
    filter_1d = expit(steepness * (np.abs(freqs) - k0))
    return filter_1d

def tomografia(projections, angles, rows, use_filter=False):
    reconstruccion = np.zeros((rows, len(projections[0])))
    
    for señal, angle in zip(projections, angles):
        if use_filter:
            signal_fft = np.fft.fft(señal)
            filtro = pasaaltas(len(señal))
            signal_fft *= filtro
            señal = np.fft.ifft(signal_fft).real
        
        imagen_rotada = ndi.rotate(
            np.tile(señal[:,None], rows).T,
            angle,
            reshape=False,
            mode="reflect"
        )
        reconstruccion += imagen_rotada
    
    return reconstruccion

projections = np.load('tomography_data/1.npy')
n_proj = len(projections)
angles = np.linspace(0, 180, n_proj, endpoint=False)
rows = len(projections[0])

sin_filtro = tomografia(projections, angles, rows, False)
con_filtro = tomografia(projections, angles, rows, True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(sin_filtro)
axes[0].set_title('Sin filtro')
axes[1].imshow(con_filtro)
axes[1].set_title('Con filtro')
plt.savefig('3.png', dpi=150, bbox_inches='tight')