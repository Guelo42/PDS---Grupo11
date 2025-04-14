
# leer_ECG_data_graficar_FFT.py
import numpy as np
import matplotlib.pyplot as plt
import peakutils

""""
Este programa lee un archivo con nombre "f{vocal}.txt". Es decir el archivo .wav previamente convertido a txt con los datos de la señal.
Mostrara un grafico con la forma de onda de la señal y otro grafico con la FFT de la señal.
"""
vocals = ["A", "E", "I", "O", "U"]  # Cambia a "A", "E", "I", "O", "U" según sea necesario
samples = "Miguel"
carpeta = f"datos/{samples}"



for vocal in vocals:
    
    # Ruta del archivo ASCII (archivo generado previamente)
    file_path = rf"{carpeta}/{vocal}.txt"
    # Cargar los datos del archivo, ignorando la primera fila
    data = np.loadtxt(file_path, skiprows=1)

    # Leer todas las líneas del archivo primero para buscar el valor de la frecuencia de muestreo
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Buscar la línea que contiene la frecuencia de muestreo (ejemplo: "# Frecuencia de muestreo: 500 Hz")
    fs_line = None
    for line in lines:
        if 'Frecuencia de muestreo' in line:
            fs_line = line
            break

    # Extraer el valor de la frecuencia de muestreo
    if fs_line:
        # Asumimos que el formato es algo como "# Frecuencia de muestreo: 500 Hz"
        fs_value = fs_line.split(":")[1].strip()  # Extraer la parte después de ":"
        fs = int(fs_value.split()[0])  # Obtener el valor numérico (asumimos que es un número entero)
        print(f"Frecuencia de muestreo extraída: {fs} Hz")
    else:
        raise ValueError("No se encontró la frecuencia de muestreo en el archivo.")

    # Ahora cargar los datos numéricos
    # Usar np.loadtxt para leer los datos a partir de la línea que no contiene encabezados
    data = np.loadtxt(file_path, comments='#')
    t = []
    # Separar el tiempo y la amplitud
    for i, value in enumerate(data):
        t.append(i*1/fs) 
    
    # Graficar la forma de onda 
    plt.figure(figsize=(10, 4))
    plt.plot(t, data, label='Forma de onda audio')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title(f'Forma de onda audio_{vocal}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{carpeta}/signal_{vocal}.png")
    #plt.show()

    # Calcular la FFT de la señal
    n = len(data)  # Número de muestras
    frecuencia = np.fft.fftfreq(n, d=1/fs)  # Vector de frecuencias
    fft_signal = np.fft.fft(data)  # FFT de la señal

    # Tomar solo las frecuencias positivas (simétricas)
    positive_freqs = frecuencia[:n // 2]
    positive_fft = np.abs(fft_signal[:n // 2])


    # Graficar la FFT con barras (sin interpolación)
    plt.figure(figsize=(10, 4))
    plt.bar(positive_freqs, positive_fft, width=0.5, color='blue')  # Usamos barras en lugar de líneas
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.title(f'Espectro de Frecuencia_{vocal}')
    plt.grid(True)
    plt.xlim(-10, fs/2)
    plt.savefig(f"{carpeta}/espectro_{vocal}.png")
    plt.show()















