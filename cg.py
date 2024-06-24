import numpy as np
import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
import sys


# Definir a codificação com base no sistema operacional
encoding = 'utf-8' if sys.platform != 'win32' else 'mbcs'

# Algoritmo tradicional de rasterização de linha (Bresenham)
def bresenham(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

# Define uma rede neural simples para suavização
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(None, None, 1)))  # Usar Input como a primeira camada
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(1, (3, 3), padding='same'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Função para suavizar a imagem usando a rede neural
def smooth_line_with_cnn(image, points, model):
    for point in points:
        image[point[1], point[0]] = 255
    image = image.astype('float32') / 255.0
    image_tensor = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
    smoothed_tensor = model.predict_on_batch(image_tensor)
    smoothed_image = (smoothed_tensor.squeeze() * 255.0).astype('uint8')
    return smoothed_image

# Função para medir tempo e memória
def measure_performance(func, *args):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end_time - start_time, peak / 10**6  # memória em MB

# Exemplo de uso
img_size = 1000
img = np.zeros((img_size, img_size), dtype=np.uint8)

# Medir desempenho do algoritmo tradicional
points, time_bresenham, memory_bresenham = measure_performance(bresenham, 10, 10, 800, 800)

# Desenhar a linha tradicional na imagem
for point in points:
    img[point[1], point[0]] = 255

# Criar e treinar a rede neural para suavização
model = create_model()
# Treinamento da rede (simulação de treinamento)
# Aqui, para simplificação, consideramos que o modelo já foi treinado.

# Medir desempenho da melhoria com IA
smoothed_image, time_smooth, memory_smooth = measure_performance(smooth_line_with_cnn, img.copy(), points, model)

# Qualidade da imagem (simples comparação de intensidade)
quality_bresenham = np.sum(img)
quality_smooth = np.sum(smoothed_image)

# Criar tabela comparativa
data = {
    "Algoritmo": ["Bresenham", "IA Suavizada"],
    "Tempo (s)": [time_bresenham, time_smooth],
    "Memoria (MB)": [memory_bresenham, memory_smooth],
    "Qualidade (Intensidade)": [quality_bresenham, quality_smooth]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

# Plotar resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Bresenham")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("IA Suavizada")
plt.imshow(smoothed_image, cmap='gray')

plt.show()
