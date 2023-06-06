import numpy as np
import skfuzzy as fuzz

# Definir los rangos y conjuntos difusos para las variables de entrada y salida
palabras = np.arange(0, 11, 1)
puntuacion = np.arange(0, 11, 1)
spam = np.arange(0, 11, 1)

# Definir las funciones de pertenencia difusa para cada conjunto difuso
palabras_baja = fuzz.trimf(palabras, [0, 0, 5])
palabras_media = fuzz.trimf(palabras, [0, 5, 10])
palabras_alta = fuzz.trimf(palabras, [5, 10, 10])

puntuacion_baja = fuzz.trimf(puntuacion, [0, 0, 5])
puntuacion_media = fuzz.trimf(puntuacion, [0, 5, 10])
puntuacion_alta = fuzz.trimf(puntuacion, [5, 10, 10])

spam_no = fuzz.trimf(spam, [0, 0, 5])
spam_si = fuzz.trimf(spam, [0, 5, 10])

# Generar las reglas difusas
regla1 = np.fmin(palabras_baja, puntuacion_baja)
regla2 = np.fmin(palabras_media, puntuacion_baja)
regla3 = np.fmin(palabras_alta, puntuacion_baja)
regla4 = np.fmin(palabras_baja, puntuacion_media)
regla5 = np.fmin(palabras_media, puntuacion_media)
regla6 = np.fmin(palabras_alta, puntuacion_media)
regla7 = np.fmin(palabras_baja, puntuacion_alta)
regla8 = np.fmin(palabras_media, puntuacion_alta)
regla9 = np.fmin(palabras_alta, puntuacion_alta)
regla10 = np.fmin(palabras_baja, puntuacion_alta)
regla11 = np.fmin(palabras_media, puntuacion_media)
regla12 = np.fmin(palabras_alta, puntuacion_baja)
regla13 = np.fmin(palabras_media, puntuacion_alta)
regla14 = np.fmin(palabras_alta, puntuacion_alta)

# Combinar las reglas
agregado = np.fmax(
    np.fmax(np.fmax(np.fmax(np.fmax(regla1, regla2), np.fmax(regla3, regla4)), np.fmax(regla5, regla6)), np.fmax(regla7, regla8)),
    np.fmax(np.fmax(np.fmax(np.fmax(regla9, regla10), regla11), regla12), np.fmax(regla13, regla14))
)

# Calcular el resultado difuso
def clasificar_spam(texto):
    # Calcular los valores difusos para palabras y puntuación (aquí puedes ajustar los valores según tu criterio)
    palabras_valor = fuzz.interp_membership(palabras, palabras_media, len(texto.split()))
    puntuacion_valor = fuzz.interp_membership(puntuacion, puntuacion_media, 8)

    # Evaluar las reglas difusas con los valores de entrada
    activacion_reglas = np.fmin(palabras_valor, puntuacion_valor)

    # Combinar las reglas activadas
    agregado_activado = np.fmin(activacion_reglas, agregado)

    # Calcular el resultado difuso
    resultado = fuzz.defuzz(spam, agregado_activado, 'centroid')
    resultado_spam = fuzz.interp_membership(spam, agregado_activado, resultado)

    return resultado_spam

# Ejemplo de uso
texto_1 = "Hola, ¿cómo estás?"


# Clasificar los textos como spam o no spam
resultado_1 = clasificar_spam(texto_1)

# Determinar si los textos son spam o no
if resultado_1 > 0.5:
    print("El texto 1 es spam.")
else:
    print("El texto 1 no es spam.")