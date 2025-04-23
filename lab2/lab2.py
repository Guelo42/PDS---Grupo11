import numpy as np
import matplotlib.pyplot as plt

def graficar_fft(signal, fs, title='FFT de la señal'):
    """
    Calcula y grafica la FFT de una señal dada.

    Parámetros:
    - signal: array de muestras de la señal.
    - fs: frecuencia de muestreo en Hz.
    - title: título del gráfico.
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result) / N
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Solo se muestra la mitad positiva del espectro
    half = N // 2
    plt.figure(figsize=(12, 4))

    # Magnitud
    plt.subplot(1, 2, 1)
    plt.stem(freqs[:half], fft_magnitude[:half], basefmt=" ")
    plt.title(f'Magnitud - {title}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('|X(f)|')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


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

def senSignal(f, n0, N, fs, phi):
    n = np.arange(n0, n0 + N)
    signal = np.sin(2 * np.pi * f * n/fs + phi)
    return signal

signals = [
    senSignal(40, 0, 120, 1000, 0),
    senSignal(40, 0, 120, 1200, 0),
    senSignal(40, 0, 120, 1300, 0),
    senSignal(60, 0, 120, 1000, np.pi/2),
    senSignal(60, 0, 120, 1200, np.pi/2),
    senSignal(1040, 0, 120, 1000, 0),
]

x7 = signals[1] + signals[4] #Señal x7 = x1 + x4
x8 = signals[2] + signals[5] #Señal x7 = x1 + x4

T7, value = esPeriodica(x7, 0)
T8, value = esPeriodica(x8, 0.1)

print(f"Señal x7: {T7} muestras")
print(f"Señal x8: {T8} muestras")

# graficar_fft(x7, 60, title='FFT de la señal x7')
# graficar_fft(x7, 325, title='FFT de la señal x8')

# ----------------------- PUNTO 2 -----------------------
# graficar_fft(x7, 120, title='FFT de la señal x7 con T=2.N')

# ----------------------- PUNTO 3 -----------------------
def intercalar_ceros(signal):
    """
    Dada una señal discreta, genera una nueva señal x9[n]
    intercalando un cero entre cada muestra de la original.

    Parámetros:
    - signal: array-like, señal original (x7[n])

    Retorna:
    - x9: numpy array, señal con ceros intercalados
    """
    N = len(signal)
    x9 = np.zeros(2 * N)
    x9[::2] = signal  # Asigna valores solo en posiciones pares
    return x9

x9 = intercalar_ceros(x7)
graficar_fft(x9, 60, title='FFT de la señal x9')

# ----------------------- PUNTO 4 -----------------------
def obtener_x10_x11(signal):
    x10 = signal[::2]  # Muestras en posiciones pares
    x11 = signal[1::2]  # Muestras en posiciones impares
    return x10, x11

x10, x11 = obtener_x10_x11(x7)

# Graficamos ambas señales
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.stem(x10, basefmt=" ")
plt.title('x10[n] = x7[2n]')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(x11, basefmt=" ")
plt.title('x11[n] = x7[2n+1]')
plt.grid(True)

plt.tight_layout()
# plt.show()


def verificar_combinacion_fft(x7):
    """
    Verifica que X7[k] = (X10[k] + exp(-j*2*pi*k/N) * X11[k]) / 2

    Parámetros:
    - x7: señal periódica original (x7[n])

    Muestra gráficos de comparación.
    """
    N = len(x7)  # asumimos que x7 tiene un período completo

    # Paso 1: obtener x10[n] y x11[n]
    x10 = x7[::2]
    x11 = x7[1::2]

    # Paso 2: calcular las FFTs (DTFS)
    X7 = np.fft.fft(x7) / N
    X10 = np.fft.fft(x10, n=N) / (N // 2)
    X11 = np.fft.fft(x11, n=N) / (N // 2)

    # Paso 3: reconstruir X7[k] con la fórmula dada
    k = np.arange(N)
    W = np.exp(-1j * 2 * np.pi * k / N)
    X7_reconstruida = (X10 + W * X11) / 2

    # Paso 4: comparar gráficamente
    plt.figure(figsize=(12, 4))

    # Magnitud
    plt.subplot(1, 2, 1)
    plt.stem(k, np.abs(X7), linefmt='b-', markerfmt='bo', basefmt=' ', label='|X7[k]| original')
    plt.stem(k, np.abs(X7_reconstruida), linefmt='r--', markerfmt='rx', basefmt=' ', label='|X7[k]| reconstruida')
    plt.title('Magnitud de X7[k]')
    plt.xlabel('k')
    plt.ylabel('|X[k]|')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
# verificar_combinacion_fft(x7[:T7])

# ----------------------- PUNTO 5 -----------------------

def sistema_a(x):
    """
    Implementa el sistema a) como convolución lineal.

    y[n] = 7.29 x[n] - 8.29 x[n-1] - 4.01 x[n-2] + 5.83 x[n-3]
    """
    h = np.array([7.29, -8.29, -4.01, 5.83])  # respuesta al impulso
    y = np.convolve(x, h, mode='full')[:len(x)]  # truncamos para tener misma longitud
    return y

def sistema_b(x):
    """
    Implementa el sistema b) usando una ecuación en diferencias con retroalimentación.
    """
    y = np.zeros_like(x)
    for n in range(len(x)):
        x_n = x[n] if n >= 0 else 0
        x_n1 = x[n-1] if n-1 >= 0 else 0
        x_n2 = x[n-2] if n-2 >= 0 else 0
        y_n1 = y[n-1] if n-1 >= 0 else 0
        y_n2 = y[n-2] if n-2 >= 0 else 0

        y[n] = (1.00025 * x_n
                - 1.86001 * x_n1
                + 1.00025 * x_n2
                + 1.80113 * y_n1
                - 0.93816 * y_n2)
    return y

y_a = sistema_a(x7[:T7])  # respuesta del sistema (a)
y_b = sistema_b(x7[:T7])  # respuesta del sistema (b)

T_a, es_per_a = esPeriodica(y_a, eps=0.01)
T_b, es_per_b = esPeriodica(y_b, eps=0.01)

print(f"Sistema (a): Periódica = {es_per_a}, con periodo T = {T_a}")
print(f"Sistema (b): Periódica = {es_per_b}, con periodo T = {T_b}")

def expandir_intervalo(signal, n_min, n_max):
    longitud = n_max - n_min + 1
    signal_expandida = np.zeros(longitud)
    offset = -n_min
    for i in range(len(signal)):
        if 0 <= i + offset < longitud:
            signal_expandida[i + offset] = signal[i]
    return signal_expandida

x7_intervalo = expandir_intervalo(x7[:T7], -18, 96)
y_a_intervalo = sistema_a(x7_intervalo)
y_b_intervalo = sistema_b(x7_intervalo)

T_ai, es_per_ai = esPeriodica(y_a_intervalo, eps=0.01)
T_bi, es_per_bi = esPeriodica(y_b_intervalo, eps=0.01)

print(f"(a) en intervalo: Periódica = {es_per_ai}, T = {T_ai}")
print(f"(b) en intervalo: Periódica = {es_per_bi}, T = {T_bi}")

