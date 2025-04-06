import wave
import numpy as np
import os  # Importar módulo para eliminar archivos

""""
Este programa lee un archivo con nombre "f{vocal}.wav" donde {vocal} es la vocal que se quiere analizar. 
La variable "vocal" se puede cambiar a "A", "E", "I", "O", "U" según sea necesario.
La salida sera un archivo de texto con el nombre "f{vocal}.txt" que contiene la frecuencia de muestreo y los datos de la señal.

Por ejemplo, si se quiere analizar la vocal "A". Se escribe "A" en la variable vocal. El archivo de entrada sera "A.wav" y el archivo de salida sera "A.txt".
"""

# Definir la vocal (cambiar aquí para procesar otra vocal)
vocal = "A"  # Cambia a "A", "E", "I", "O", "U" según sea necesario

carpeta = "tp1/datos/muestrasMiguel"  # Cambia a la carpeta donde están los archivos .wav
# Rutas de los archivos basadas en la vocal
wav_file_path = f"{carpeta}/{vocal}.wav"
txt_file_path = f"{carpeta}/{vocal}.txt"
temp_file_path = f"temp_{vocal}.txt"

# Cargar el archivo de audio WAV
with wave.open(wav_file_path, "rb") as wav_file:
    sample_rate = wav_file.getframerate()  # Frecuencia de muestreo original
    num_frames = wav_file.getnframes()
    audio_data = wav_file.readframes(num_frames)

# Convertir a array NumPy
samples = np.frombuffer(audio_data, dtype=np.int16)

# Guardar los datos en un archivo de texto temporal
np.savetxt(temp_file_path, samples, fmt="%.6f")

# Crear el archivo final con la línea de frecuencia de muestreo
with open(txt_file_path, "w", encoding="utf-8") as final_file:
    final_file.write(f"# Frecuencia de muestreo: {sample_rate} Hz\n")  # Agregar encabezado
    with open(temp_file_path, "r", encoding="utf-8") as temp:
        final_file.write(temp.read())  # Escribir el resto del archivo

# Eliminar el archivo temporal
os.remove(temp_file_path)

print(f"Archivo guardado con {num_frames} muestras a {sample_rate} Hz en {txt_file_path}")
