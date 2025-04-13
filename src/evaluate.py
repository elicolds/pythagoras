import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    # Definir rutas base para dataset y modelo
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    dataset_path = os.path.join(base_path, 'data', 'triangle_dataset.csv')
    model_path = os.path.join(base_path, 'models', 'linear_regression_model.pkl')
    
    # Cargar dataset y separar variables
    data = pd.read_csv(dataset_path)
    X = data[['a', 'b']]
    y = data['c']
    
    # Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # Hacer predicciones
    predictions = model.predict(X)
    
    # Calcular métricas de rendimiento
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
    print(f"R² (Coeficiente de Determinación): {r2:.4f}")
    
    # Crear gráfico de predicción vs. valores reales
    plt.figure(figsize=(8,6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicción vs Real (Regresión Lineal)")
    
    # Agregar una línea de identidad (predicción ideal)
    min_val = min(min(y), min(predictions))
    max_val = max(max(y), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Ideal')
    plt.legend()
    
    # Guardar el gráfico en la carpeta results
    output_path = os.path.join(base_path, 'results', 'pred_vs_true.png')
    plt.savefig(output_path)
    print(f"Gráfico guardado en {output_path}")

if __name__ == "__main__":
    evaluate_model()
