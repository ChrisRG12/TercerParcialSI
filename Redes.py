import tensorflow as tf

import numpy as np

# Definimos los parámetros de la red neuronal
tamaño_entrada = 784  # tamaño de entrada de la imagen MNIST (28x28 píxeles = 784)
tamaño_oculto = 128  # tamaño de la capa oculta
tamaño_salida = tamaño_entrada  # tamaño de salida es igual al tamaño de entrada

# Definimos la entrada del modelo
entrada = tf.keras.layers.Input(shape=(tamaño_entrada,))

# Definimos la capa oculta de la red neuronal
oculta = tf.keras.layers.Dense(tamaño_oculto, activation='relu')(entrada)

# Definimos la capa de salida de la red neuronal
salida = tf.keras.layers.Dense(tamaño_salida, activation='sigmoid')(oculta)

# Creamos el modelo completo
autoencoder = tf.keras.models.Model(entrada, salida)

# Compilamos el modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Cargamos los datos de MNIST
(x_entrenamiento, _), (x_prueba, _) = tf.keras.datasets.mnist.load_data()

# Normalizamos los datos
x_entrenamiento = x_entrenamiento.astype('float32') / 255.
x_prueba = x_prueba.astype('float32') / 255.

# Aplanamos los datos de entrada
x_entrenamiento = np.reshape(x_entrenamiento, (len(x_entrenamiento), tamaño_entrada))
x_prueba = np.reshape(x_prueba, (len(x_prueba), tamaño_entrada))

# Entrenamos el modelo
autoencoder.fit(x_entrenamiento, x_entrenamiento,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_prueba, x_prueba))

# Usamos el modelo entrenado para codificar y decodificar una imagen de prueba
imagen_codificada = autoencoder.predict(x_prueba)
imagen_decodificada = autoencoder.predict(imagen_codificada)
