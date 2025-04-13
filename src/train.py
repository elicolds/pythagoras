import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_linear_regression():
    # Definir la ruta del dataset
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    dataset_path = os.path.join(base_path, 'data', 'triangle_dataset.csv')
    
    # Cargar el dataset
    data = pd.read_csv(dataset_path)
    X = data[['a', 'b']]   # Características: a y b
    y = data['c']          # Objetivo: c (hipotenusa)
    
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Asegurarse de que la carpeta models exista
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar el modelo entrenado usando joblib
    model_path = os.path.join(models_dir, 'linear_regression_model.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo de Regresión Lineal entrenado y guardado en {model_path}")

if __name__ == "__main__":
    train_linear_regression()
