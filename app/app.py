import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# 4.1. Componentes del sistema difuso
# Variables de entrada
distancia_al_hoyo = ctrl.Antecedent(np.arange(0, 100, 1), 'distancia_al_hoyo')
velocidad_viento = ctrl.Antecedent(np.arange(0, 40, 1), 'velocidad_viento')

# Variable de salida
golpe = ctrl.Consequent(np.arange(0, 100, 1), 'golpe')

# Conjuntos difusos
distancia_al_hoyo['corta'] = fuzz.trimf(distancia_al_hoyo.universe, [0, 0, 30])
distancia_al_hoyo['media'] = fuzz.trimf(distancia_al_hoyo.universe, [20, 50, 80])
distancia_al_hoyo['larga'] = fuzz.trimf(distancia_al_hoyo.universe, [70, 100, 100])

velocidad_viento['baja'] = fuzz.trimf(velocidad_viento.universe, [0, 0, 20])
velocidad_viento['media'] = fuzz.trimf(velocidad_viento.universe, [10, 20, 30])
velocidad_viento['alta'] = fuzz.trimf(velocidad_viento.universe, [20, 40, 40])

golpe['suave'] = fuzz.trimf(golpe.universe, [0, 0, 50])
golpe['medio'] = fuzz.trimf(golpe.universe, [30, 50, 70])
golpe['fuerte'] = fuzz.trimf(golpe.universe, [50, 100, 100])

# Reglas difusas
rule1 = ctrl.Rule(distancia_al_hoyo['corta'] & velocidad_viento['baja'], golpe['suave'])
rule2 = ctrl.Rule(distancia_al_hoyo['media'] & velocidad_viento['media'], golpe['medio'])
rule3 = ctrl.Rule(distancia_al_hoyo['larga'] & velocidad_viento['alta'], golpe['fuerte'])

# 4.2. Esquema de funcionamiento del simulador
simulador_golf = ctrl.ControlSystem([rule1, rule2, rule3])
simulador = ctrl.ControlSystemSimulation(simulador_golf)

# 4.3. Simulación en 2D
distancia_al_hoyo_input = 40  # Ejemplo de entrada
velocidad_viento_input = 15  # Ejemplo de entrada

simulador.input['distancia_al_hoyo'] = distancia_al_hoyo_input
simulador.input['velocidad_viento'] = velocidad_viento_input

simulador.compute()

golpe_output = simulador.output['golpe']
print(f"Golpe: {golpe_output:.2f}")

# 4.4. Simulación en 3D (Movimiento parabólico)
tiempo_total_vuelo = 2 * distancia_al_hoyo_input / (golpe_output * np.cos(np.radians(45)))
tiempo = np.linspace(0, tiempo_total_vuelo, num=100)

x = distancia_al_hoyo_input * np.cos(np.radians(45)) * tiempo
y = distancia_al_hoyo_input * np.sin(np.radians(45)) * tiempo - 0.5 * 9.8 * tiempo**2

# Visualización en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, tiempo)

# Configuración de la gráfica
ax.set_xlabel('Distancia (m)')
ax.set_ylabel('Altura (m)')
ax.set_zlabel('Tiempo (s)')
ax.set_title('Simulación de golpe en 3D con movimiento parabólico')

plt.show()

# 4.3. Simulación en 2D
tiempo_total_vuelo = 2 * distancia_al_hoyo_input / (golpe_output * np.cos(np.radians(45)))
tiempo = np.linspace(0, tiempo_total_vuelo, num=100)

x = distancia_al_hoyo_input * np.cos(np.radians(45)) * tiempo
y = distancia_al_hoyo_input * np.sin(np.radians(45)) * tiempo - 0.5 * 9.8 * tiempo**2

# Visualización en 2D
plt.plot(x, y)
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.title('Simulación de golpe en 2D con movimiento parabólico')
plt.grid(True)
plt.show()



# 4.5. Uso de un algoritmo genético para calcular los parámetros adecuados para el golpe

# Función de aptitud
def calcular_aptitud(parametros):
    distancia_al_hoyo = realizar_simulacion(parametros)
    aptitud = 1 / distancia_al_hoyo

    return aptitud

# Función de simulación del golpe
def realizar_simulacion(parametros):
    distancia_al_hoyo = np.random.randint(50, 150)  # Distancia aleatoria al hoyo

    return distancia_al_hoyo

# Algoritmo genético
def algoritmo_genetico():
    # Parámetros del algoritmo genético
    tamano_poblacion = 50
    tamano_cromosoma = 2
    num_generaciones = 100
    probabilidad_cruce = 0.8
    probabilidad_mutacion = 0.1

    # Inicializar población aleatoria
    poblacion = []
    for _ in range(tamano_poblacion):
        cromosoma = [random.uniform(0, 100), random.uniform(0, 90)]  # Valores aleatorios de parámetros (fuerza y ángulo)
        poblacion.append(cromosoma)

    # Evolución de la población
    for _ in range(num_generaciones):
        # Calcular aptitud de la población
        aptitudes = [calcular_aptitud(cromosoma) for cromosoma in poblacion]

        # Selección de padres
        padres = random.choices(poblacion, weights=aptitudes, k=tamano_poblacion)

        # Nueva generación
        nueva_generacion = []
        while len(nueva_generacion) < tamano_poblacion:
            # Cruce
            padre1, padre2 = random.sample(padres, k=2)
            if random.random() < probabilidad_cruce:
                hijo = [padre1[0], padre2[1]]  # Cruce de los parámetros
            else:
                hijo = padre1

            # Mutación
            if random.random() < probabilidad_mutacion:
                hijo[random.randint(0, tamano_cromosoma - 1)] = random.uniform(0, 100)  # Mutación de un parámetro

            nueva_generacion.append(hijo)

        # Actualizar población
        poblacion = nueva_generacion

    # Obtener la mejor solución (mejor cromosoma)
    mejores_aptitudes = [calcular_aptitud(cromosoma) for cromosoma in poblacion]
    indice_mejor = np.argmax(mejores_aptitudes)
    mejor_solucion = poblacion[indice_mejor]

    return mejor_solucion


mejor_parametro = algoritmo_genetico()
print("Mejor parámetro encontrado:", mejor_parametro)
