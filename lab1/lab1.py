import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

carpeta = "lab1/datos"  #Carpeta donde se guardan los archivos .wav y .npz

### ------------------------ ACTIVIDAD 1 ----------------------------------

def esPeriodica(signal, eps):
    N = len(signal) #Numero de muestras
    T = 1 #Periodo de la señal
    for T in range(1, N//2 + 1):
        for n in range(0, N-T):
            if np.abs(signal[n] - signal[n+T]) > eps + 1e-10: #eps agregado por errores introducidos por los calculos en punto flotante 
                break
            elif n == N-T-1:
                return T, True
        if T == N//2:
            return 0, False



#### ------------------------ ACTIVIDAD 2 ----------------------------------

def senSignal(f, n0, N, fs, phi):
    n = np.arange(n0, n0 + N)
    signal = np.sin(2 * np.pi * f * n/fs + phi)
    return signal


### ------------------------ ACTIVIDAD 3 ----------------------------------

def saveSignal(nombre, signal, fs, n):
    wavfile.write(f"{carpeta}/{nombre}.wav", int(fs), (signal * 32767).astype(np.int16))
    np.savez(f"{carpeta}/{nombre}.npz", signal=signal, fs=fs, n=n)

def graph(n, signal, nombre):
    plt.figure()
    plt.stem(n, signal)
    plt.title(f"Señal {nombre}")
    plt.xlabel("n")
    plt.ylabel(f"{nombre}[n]")
    plt.grid(True)
    plt.show()
    ##plt.savefig(f"{nombre}.png")
    plt.close()


signals = [
    senSignal(40, -20, 120, 1000, 0),
    senSignal(40, -20, 120, 1200, 0),
    senSignal(40, -20, 120, 1300, 0),
    senSignal(60, -20, 120, 1000, np.pi/2),
    senSignal(60, -20, 120, 1200, np.pi/2),
    senSignal(1040, -20, 120, 1000, 0),
]

signals.append(signals[1] + signals[4]) #Señal x7 = x1 + x4
signals.append(signals[2] + signals[5]) #Señal x8 = x2 + x5

signals = [
    senSignal(40, -20, 120, 1000, 0),
    senSignal(40, -20, 120, 1200, 0),
    senSignal(40, -20, 120, 1300, 0),
    senSignal(60, -20, 120, 1000, np.pi/2),
    senSignal(60, -20, 120, 1200, np.pi/2),
    senSignal(1040, -20, 120, 1000, 0),
]

fs_values = [1000, 1200, 1300, 1000, 1200, 1000]
eps_values = [0, 0, 0.1, 0.2, 0, 0, 0, 0, 0]    #Si pongo eps4 en 0, el calculo del periodo da 50n porque el periodo no es multiplo del tiempo de muestreo.
#Si pongo eps3 en 0 no se detecta periodicidad por el mismo motivo

signals.append(signals[1] + signals[4])  # Señal x7 = x1 + x4
fs_values.append(1000)
signals.append(signals[2] + signals[5])  # Señal x8 = x2 + x5
fs_values.append(1200)

for i, (signal, fs, eps) in enumerate(zip(signals, fs_values, eps_values), start=1):

    T, value = esPeriodica(signal, eps)
    if(value):
        print(f"Periodo x{i} = {T}n")
    else:
        print(f"x{i} no es periodica")

    graph(np.arange(-20, 100), signal, f"x{i}")
    saveSignal(f"x{i}", signal, fs, np.arange(-20, 100))

