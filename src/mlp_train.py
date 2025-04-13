import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
import joblib

def train_mlp():
    # Definir la ruta base y la del dataset
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    dataset_path = os.path.join(base_path, 'data', 'triangle_dataset.csv')
    
    # Cargar el dataset
    data = pd.read_csv(dataset_path)
    X = data[['a', 'b']]
    y = data['c']
    
    # Crear y entrenar el modelo MLPRegressor
    # Los parámetros son ajustables según sea necesario
    mlp_model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam',
                             max_iter=1000, random_state=42)
    mlp_model.fit(X, y)
    
    # Asegurarse de que la carpeta models exista
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar el modelo entrenado
    model_path = os.path.join(models_dir, 'mlp_model.pkl')
    joblib.dump(mlp_model, model_path)
    print(f"Modelo MLP entrenado y guardado en {model_path}")

if __name__ == '__main__':
    train_mlp()
