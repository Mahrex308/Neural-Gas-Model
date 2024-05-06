# Librerias
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np


# Modelo Gas Neural
class ModeloGasNeuronal:
    def __init__(self, X, n_particulas, puntos_gen_mezcla=1):
        # Numero de partículas en el modelo
        self.n_particulas = n_particulas  
        # Dimensionalidad de los datos
        self.dimension_dominio = X.shape[1]  
        # Inicializacion de las posiciones de las mismas particulas
        self.particulas = np.zeros((n_particulas, self.dimension_dominio))  
        for i in range(n_particulas):
            # Se establecen las posiciones iniciales de las partículas
            self.particulas[i, :] = generador_particulas_mezcla(X, puntos_gen_mezcla)  

    def ajustar(self, X, iteraciones, lr0=1, vecindad0='auto', uprate=0.5, delta_coef_min_actualizacion=0):
        # Numero de muestras de entrenamiento
        num_muestras = X.shape[0] 
        if vecindad0 == 'auto':
            # Establece de manera automatica el tamaño de la vecindad con respecto al tamaño (se agrega despues)
            vecindad0 = max(self.n_particulas // 10, 2)  
        for i in range(iteraciones):
            # Selecciona un punto de datos al azar dentro del array de datos
            punto_elegido = X[np.random.choice(num_muestras), :]  
            # Calcula las distancias entre las particulas (con distancia euclidiana)
            distancias_particulas = euclidean_distances([punto_elegido], self.particulas)[0]  
            # Ordena todas las particulas por distancia
            indices_particulas_ordenados = np.argsort(distancias_particulas)  
            # Numero de particulas a actualizar
            num_particulas_actualizadas = int(uprate * len(indices_particulas_ordenados))  
            for rango, indice_particula in enumerate(indices_particulas_ordenados[:num_particulas_actualizadas]):
                # Calcula el coeficiente de actualizacion
                delta_coef = lr0 * ((iteraciones - i) / iteraciones) * np.exp(-(2 * rango) / (vecindad0 * ((iteraciones - i) / iteraciones) ** 2))  
                if delta_coef > delta_coef_min_actualizacion:
                    # Actualiza la posicion de la particula
                    self.particulas[indice_particula] += delta_coef * (punto_elegido - self.particulas[indice_particula])  


def generador_particulas_mezcla(X, n_puntos):
    # Muestrea puntos de datos para cada particula
    indices = np.random.choice(X.shape[0], n_puntos, replace=False)  
    # Devuelve posiciones iniciales de las particulas
    return np.mean(X[indices, :], axis=0)  


# -- DATOS DE ENTRADA --
# Numero de puntos de datos en forma de anillo para generar
num_puntos_datos = 40000  
# Numero de unidades en el modelo Gas Neuronal
num_unidades = 400  


# -- ITERACIONES A MOSTRAR EN EL PLOTEO FINAL --
# Iteraciones de entrenamiento que se mostraran
iteraciones = [0, 300, 2500, 40000] 


# -- CIRCUNFERENCIA CON RESPECTO A LA GENERACION DE DATOS DENTRO DE LA FORMA DE ANILLO -- 
# Angulos aleatorios
angulos = np.random.rand(num_puntos_datos) * 2 * np.pi 
# Radios aleatorios 
radio = 0.4 + np.random.rand(num_puntos_datos) * 0.2  
# Genera puntos de datos en forma de anillo
X = np.column_stack((0.5 + radio * np.cos(angulos), 0.5 + radio * np.sin(angulos)))  


# Llama y crea el modelo Gas Neuronal con respecto la clase anterior
modelo = ModeloGasNeuronal(X, num_unidades)  


# -- CREAMOS Y DECLARAMOS EL TAMAÑO QUE TENDRA NUESTRA GRAFICAS CON LAS SALIDAS REQUERIDAS --
# Crea subgraficos para instantaneas en una imagen de 2x2 con los 4 ploteos que se requieren 
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  
axs = axs.flatten()

for i, itr in enumerate(iteraciones):
    # Se entrena el modelo
    modelo.ajustar(X, itr)  
    # Se grafican los puntos de datos
    axs[i].scatter(X[:, 0], X[:, 1], s=10, label=' ', c='white')  
    # Se grafica las unidades del modelo
    axs[i].scatter(modelo.particulas[:, 0], modelo.particulas[:, 1], s=10, c='orange', label=' ')  
    axs[i].set_title(f'Iteración: {itr}')  
    axs[i].set_xlabel('x1')  
    axs[i].set_ylabel('x2')   

    # Parametros del anillo
    centro = (0.5, 0.5)  # Centro del anillo
    radio_interior = 0.39  # Radio interior del anillo
    radio_exterior = 0.61  # Radio exterior del anillo
    circulo_exterior = plt.Circle(centro, radio_exterior, color='blue', fill=True, alpha=0.3)  
    circulo_interior = plt.Circle(centro, radio_interior, color='white', fill=True)

    # Se añaden los circulos al gráfico (anillo)
    axs[i].add_artist(circulo_exterior)
    axs[i].add_artist(circulo_interior)

    # axs[i].legend()  


# Muestra el grafico con los datos requeridos de las 4 diferentes iteraciones de la prueba
# En este caso se mostraran las iteracion 0 o inicial, 300, 2500 y por ultimo la iteracion 40000
plt.tight_layout()  
plt.show()  