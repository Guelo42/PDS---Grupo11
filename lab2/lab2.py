import numpy as np
import matplotlib.pyplot as plt


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

##Señales
x = [
    0,
    senSignal(40, 0, 120, 1000, 0),
    senSignal(40, 0, 120, 1200, 0),
    senSignal(40, 0, 120, 1300, 0),
    senSignal(60, 0, 120, 1000, np.pi/2),
    senSignal(60, 0, 120, 1200, np.pi/2),
    senSignal(1040, 0, 120, 1000, 0),
]

x.append(x[1] + x[4]) #Señal x7 = x1 + x4
x.append(x[2] + x[5]) #Señal x8 = x2 + x5



T7, value = esPeriodica(x[7], 0)
x[7] = x[7][:T7] #Limito el numero de muestras a un solo periodo (50 muestras)
print(f"Señal x7: Periodo = {T7}, Valor = {value}")


plt.figure(figsize=(10, 6))  # Cambia los valores de ancho y alto según sea necesario
plt.subplot(2, 2, 1)  # Cambia la posición del subplot según sea necesario
plt.stem(x[7])  # Reemplaza x[7] con la señal que desees graficar
plt.title('Señal x7')  # Cambia el título según sea necesario
plt.xlabel('t')  # Cambia la etiqueta del eje X
plt.ylabel('Amplitud')  # Cambia la etiqueta del eje Y
plt.grid(True)


T8, value = esPeriodica(x[8], 0)
x[8] = x[8][:T8] #Limito el numero de muestras a un solo periodo (60 muestras)
print(f"Señal x8: Periodo = {T8}, Valor = {value}")


plt.subplot(2, 2, 2)  # Cambia la posición del subplot según sea necesario
plt.stem(x[8])  # Reemplaza x[7] con la señal que desees graficar
plt.title('Señal x8')  # Cambia el título según sea necesario
plt.xlabel('t')  # Cambia la etiqueta del eje X
plt.ylabel('Amplitud')  # Cambia la etiqueta del eje Y
plt.grid(True)



# Calcular la FFT de la señal x7
F_x7 = np.fft.fftfreq(T7, d=1/1000)  # Vector de frecuencias, la frecuencia de muestreo para x7 es 1000 Hz
fft_x7 = np.fft.fft(x[7])  # FFT de la señal x7, tomando un periodo (60 muestras)

# Tomar solo las frecuencias positivas (simétricas)
F_x7 = F_x7[:T7 // 2]
fft_x7 = np.abs(fft_x7[:T7 // 2])
fft_x7 = fft_x7 * 2 /T7  # Normalizar la FFT

plt.subplot(2, 2, 3)
_, _, baseline = plt.stem(F_x7, fft_x7)  # Gráfica de barras verticales con un punto en la parte superior
baseline.set_visible(False)
plt.xlim(0, 100)  # Rango de frecuencias de 0 a 100 Hz
plt.title('Espectro de Frecuencia de x7')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.grid(True)

# Calcular la FFT de la señal x8
F_x8 = np.fft.fftfreq(T8, d=1/1200)  # Vector de frecuencias, la frecuencia de muestreo para x8 es 1000 Hz
fft_x8 = np.fft.fft(x[8])  # FFT de la señal x8, tomando un periodo (60 muestras)

# Tomar solo las frecuencias positivas (simétricas)
F_x8 = F_x8[:T8 // 2]
fft_x8 = np.abs(fft_x8[:T8 // 2])
fft_x8 = fft_x8 * 2 /T8  # Normalizar la FFT

plt.subplot(2, 2, 4)
_, _, baseline = plt.stem(F_x8, fft_x8)  # Gráfica de barras verticales con puntos en la parte superior
baseline.set_visible(False) 
plt.xlim(0, 100)  # Rango de frecuencias de 0 a 100 Hz
plt.title('Espectro de Frecuencia de x8')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.tight_layout()  # Ajusta el espaciado entre subgráficas
plt.grid(True)


#Graficamos la señal de tiempo continuo y su correspondiente serie de Fourier


fs = 5000  # frecuencia de muestreo
T = 0.05   # un período completo de la señal compuesta (el minimo comun multiplo de 40hz y 60hz)
t = np.arange(0, T, 1/fs)
y = np.sin(80*np.pi*t) + np.cos(120*np.pi*t)

plt.figure()
plt.plot(t, y)
plt.xlim(0, T)  # Rango de frecuencias de 0 a 100 Hz
plt.title('x[t] = sen(80*pi*t) + cos(120*pi*t)')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()  # Ajusta el espaciado entre subgráficas
##plt.show()



#-------------------Actividad 2-------------------

x[7] = x[1] + x[4]  # Señal x7 = x1 + x4
x7 = x[7][:2*T7]

# Calcular la FFT de la señal x7
F_x7 = np.fft.fftfreq(2*T7, d=1/1000)  # Vector de frecuencias, la frecuencia de muestreo para x7 es 1000 Hz
fft_x7 = np.fft.fft(x7)

# Tomar solo las frecuencias positivas (simétricas)
F_x7 = F_x7[:(2*T7) // 2]
fft_x7 = np.abs(fft_x7[:2*T7 // 2])
fft_x7 = fft_x7 * 2 /(2*T7)  # Normalizar la FFT

plt.figure()

plt.suptitle('Análisis de la Señal x7 y su Espectro de Frecuencia considerando muestras de dos periodos', fontsize=12)

plt.subplot(2, 1, 1)  # Cambia la posición del subplot según sea necesario
plt.stem(x7)  # Reemplaza x[7] con la señal que desees graficar 
plt.title('Señal x7') 
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(2, 1, 2)  # Cambia la posición del subplot según sea necesario
_, _, baseline = plt.stem(F_x7, fft_x7)  # Gráfica de barras verticales con un punto en la parte superior
baseline.set_visible(False)
plt.xlim(0, 100)  # Rango de frecuencias de 0 a 100 Hz
plt.title('Espectro de Frecuencia de x7')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()  # Ajusta el espaciado entre subgráficas
##plt.show()


#-------------------Actividad 3-------------------

#Convierto la señal de modo que las muestras impares son igual a 0 y analizo la FFT
x7[1::2] = 0
x7 = x7[:T7] #Vuelvo a considerar solo un periodo

# Calcular la FFT de la señal x7
F_x7 = np.fft.fftfreq(T7, d=1/1000)  # Vector de frecuencias, la frecuencia de muestreo para x7 es 1000 Hz
fft_x7 = np.fft.fft(x7)

# Tomar solo las frecuencias positivas (simétricas)
F_x7 = F_x7[:(T7) // 2]
fft_x7 = np.abs(fft_x7[:T7 // 2])
fft_x7 = fft_x7 * 2 /(T7)  # Normalizar la FFT


plt.figure()
plt.suptitle('Análisis de la Señal x7 y su Espectro de Frecuencia igualando a 0 muestras impares', fontsize=12)

plt.subplot(2, 1, 1)  # Cambia la posición del subplot según sea necesario
plt.stem(x7)  # Reemplaza x[7] con la señal que desees graficar 
plt.title('Señal x7') 
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(2, 1, 2)  # Cambia la posición del subplot según sea necesario
_, _, baseline = plt.stem(F_x7, fft_x7)  # Gráfica de barras verticales con un punto en la parte superior
baseline.set_visible(False)
plt.xlim(0, 100)  # Rango de frecuencias de 0 a 100 Hz
plt.title('Espectro de Frecuencia de x7')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.axhline(0.5, color='r', linestyle='--', linewidth=1, label='0.5')
plt.yticks(np.arange(0, 0.6, 0.1))  # desde 0 hasta 0.5 en pasos de 0.1
plt.tight_layout()  # Ajusta el espaciado entre subgráficas
#plt.show()


#-------------------Actividad 4-------------------

x[7] = x[1] + x[4]  # Señal x7 = x1 + x4
x7 = x[7]

x10 = x7[0::2] #X9 tiene las muestras pares de x7
x10 = x10[:T7]
x11 = x7[1::2] #X10 tiene las muestras impares de x7
x11 = x11[:T7]

# Calcular la FFT de la señal x10
fft_x10 = np.fft.fft(x10)

# Calcular la FFT de la señal x11
fft_x11 = np.fft.fft(x11)

N = T7 #Periodo fundamental de x7 en numero de muestras

k = np.arange(0, T7) #Vector de frecuencias para la señal x9

DTFS_x7 = (fft_x10[k]+np.exp(-1j*2*np.pi*k/N)*fft_x11[k])/2

x7 = x[7][:T7]
fft_x7 = np.fft.fft(x7)




plt.figure()
plt.subplot(2, 1, 1)  # Cambia la posición del subplot según sea necesario
_, _, baseline = plt.stem(k, np.abs(DTFS_x7))  # Gráfica de barras verticales con un punto en la parte superior
baseline.set_visible(False)
plt.xlim(0, N) 
plt.title('DTFS de x7 en el dominio k')
plt.xlabel('K')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2, 1, 2)  # Cambia la posición del subplot según sea necesario
_, _, baseline = plt.stem(k, np.abs(fft_x7))  # Gráfica de barras verticales con un punto en la parte superior
baseline.set_visible(False)
plt.xlim(0, N) 
plt.title('FFT x7')
plt.xlabel('K')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()  # Ajusta el espaciado entre subgráficas
plt.show()

#-------------------Actividad 5-------------------

#Implementacion de funciones

def y1(x, N):
    y = []
    for n in N:
        y.append(0)
        y[n] += 7.29 * x[n]
        if n-1 >= 0:
            y[n] -= 8.29 * x[n-1]
        if n-2 >= 0:
            y[n] -= 4.01 * x[n-2]
        if n-3 >= 0:
            y[n] += 5.83 * x[n-3]
    return y

def y2(x, N):

    y = []
    for n in N:
        y.append(0)
        y[n] += 1.00025 * x[n]
        if n-1 >= 0:
            y[n] -= 1.86001 * x[n-1]
            y[n] += 1.80113 * y[n-1]
        if n-2 >= 0:
            y[n] += 1.00025 * x[n-2]
            y[n] -= 0.93816 * y[n-2]

    return y

x7 = x[1] + x[4] #Señal x7 = x1 + x4
N = np.arange(0,len(x7))

out1 = y1(x7,N)
out2 = y2(x7,N)

plt.figure()
plt.subplot(2,2,1)
plt.plot(N, x7)
plt.title('Entrada x7')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(N, out1)
plt.title('Salida y1')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2,2,3)
plt.plot(N, out2)
plt.title('Salida y2')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()


#iii)

T_y1, value = esPeriodica(out1, 0)
print(f"Señal salida1: Periodo = {T_y1}, Valor = {value}")
T_y2, value = esPeriodica(out2, 0)
print(f"Señal salida2: Periodo = {T_y2}, Valor = {value}")


#iv) Redefino las señales de entrada

x = [
    0,
    senSignal(40, -20, 120, 1000, 0),
    senSignal(40, -20, 120, 1200, 0),
    senSignal(40, -20, 120, 1300, 0),
    senSignal(60, -20, 120, 1000, np.pi/2),
    senSignal(60, -20, 120, 1200, np.pi/2),
    senSignal(1040, -20, 120, 1000, 0),
]

x7 = x[1] + x[4] #Señal x7 = x1 + x4

out1 = y1(x7,range(len(x7)))
out2 = y2(x7,range(len(x7)))

out1 = out1[4:117]
out2 = out2[4:117] #Considero el intervalo -18 <= n <= 96

plt.figure()
plt.subplot(2,2,1)
plt.plot(N, x7)
plt.title('Entrada x7')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

N = np.arange(-16, 97) #Intervalo -18 <= n <= 96

plt.subplot(2,2,2)
plt.plot(N, out1)
plt.title('Salida y1')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2,2,3)
plt.plot(N, out2)
plt.title('Salida y2')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nConsiderando el intervalo -18 <= n <= 96:")
T_y1, value = esPeriodica(out1, 0.001)
print(f"Señal salida1: Periodo = {T_y1}, Valor = {value}")
T_y2, value = esPeriodica(out2, 0.1)
print(f"Señal salida2: Periodo = {T_y2}, Valor = {value}")

out1 = y1(x[1],range(len(x[1])))
out2 = y2(x[4],range(len(x[4])))
out3 = []
for i in range(len(x[1])):
    out3.append(out1[i] + out2[i])

out4 = y1(x[1]+x[4], range(len(x[1])))

## Verificacion de linealidad del sistema 1
N = np.arange(-20, 100)
plt.figure()
plt.subplot(2,2,1)
plt.plot(N, x[1])
plt.title('Entrada x1')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2,2,2)
plt.plot(N, out1)
plt.title('salida y1(x1)')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2,2,3)
plt.plot(N, x[4])
plt.title('Entrada x4')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.subplot(2,2,4)
plt.plot(N, out2)
plt.title('salida y1(x4)')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)


plt.tight_layout()


plt.figure()
plt.subplot(2,1,1)
plt.plot(N,out3)
plt.ylim(-5, 5)  # Establece el rango del eje vertical de -5 a 5
plt.title('salida y1(x4) + y1(x1)')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(N, out4)
plt.ylim(-5, 5)  # Establece el rango del eje vertical de -5 a 5
plt.title('salida y1(x4 + x1)')
plt.xlabel('n')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.grid(True)
plt.show()






