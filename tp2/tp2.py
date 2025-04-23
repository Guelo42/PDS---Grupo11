
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import fft, fftfreq, fftshift

# Parámetros
R = 1
C = 0.5  # Capacitancia en Faradios
RC = R*C
dt = 0.01
t = np.arange(0, 5, dt)  # Tiempo de 0 a 5 segundos

# Respuesta al impulso discreta: h(t) = (1/RC) * e^(-t/RC) * u(t)
h = (1/RC) * np.exp(-t / RC)

# Entrada: escalón unitario
x = np.ones_like(t)

# Convolución discreta
y = convolve(x, h) * dt  # Multiplicamos por dt para mantener escala de tiempo
ty = np.arange(0, len(y)) * dt  # Nuevo eje temporal para la convolución

# Respuesta en frecuencia analítica: H(f) = 1 / (1 + j2πfRC)
f = fftfreq(len(t), d=dt)
H_f = 1 / (1 + 1j * 2 * np.pi * f * RC)

f = f[0:int(len(t)/2)]  # Frecuencias positivas
H_f = H_f[0:int(len(t)/2)]

#Ancho de banda (AB)

fMagMax = np.max(H_f)  # Frecuencia de mayor amplitud

#Busqueda de frecuencia de corte donde la magnitud es 1/sqrt(2) de la maxima
fc, fcMag = 0, 0
for i in range(len(H_f)):
    if H_f[i] < fMagMax / np.sqrt(2):
        fc = f[i]
        fcMag = H_f[i]
        break

print(f"Frecuencia de corte: {fc} Hz")

# Gráficos
plt.figure(figsize=(14, 8))

# Gráfico de magnitud
plt.subplot(2, 2, 1)
plt.plot(t, h)
plt.title('Respuesta al Impulso h(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

# Gráfico de salida por convolución
plt.subplot(2, 2, 2)
plt.plot(ty, y)
plt.title('Salida del sistema y(t) = x(t) * h(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

# Gráfico de magnitud de la respuesta en frecuencia
plt.subplot(2, 2, 3)
plt.plot(f, np.abs(H_f))
plt.axvline(x=fc, color='red', linestyle='--', label=f'x = fc = {fc} Hz')
plt.axhline(y=fMagMax/np.sqrt(2), color='red', linestyle='--', label=f'y = fMagMax/sqrt(2) = {abs(fMagMax/np.sqrt(2))}')
plt.legend()
plt.title('Magnitud de H(f)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(f)|')
plt.grid(True)

# Gráfico de fase de la respuesta en frecuencia
plt.subplot(2, 2, 4)
plt.plot(f, np.angle(H_f))
plt.title('Fase de H(f)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True)

plt.tight_layout()
plt.show()











