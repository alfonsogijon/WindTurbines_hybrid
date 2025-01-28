import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# 1. Cargar el dataset California Housing
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Preprocesamiento de los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Dividir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Crear el modelo en Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# 5. Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 6. Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, verbose=1)

# 7. Guardar modelo
model.save("model.keras")
print("Modelo guardado en 'model.keras'")
