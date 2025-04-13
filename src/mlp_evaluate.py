import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_mlp_model():
    # Definir las rutas base para el dataset y el modelo
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    dataset_path = os.path.join(base_path, 'data', 'triangle_dataset.csv')
    model_path = os.path.join(base_path, 'models', 'mlp_model.pkl')
    
    # Cargar el dataset
    data = pd.read_csv(dataset_path)
    X = data[['a', 'b']]
    y = data['c']
    
    # Cargar el modelo MLP entrenado
    mlp_model = joblib.load(model_path)
    
    # Hacer las predicciones con el modelo
    predictions = mlp_model.predict(X)
    
    # Calcular las métricas de desempeño: MSE y R²
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"MSE (Error Cuadrático Medio) para MLP: {mse:.4f}")
    print(f"R² (Coeficiente de Determinación) para MLP: {r2:.4f}")
    
    # Crear gráfico: Comparación de predicciones vs. valores reales
    plt.figure(figsize=(8,6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicción vs Real (MLP Regressor)")
    
    # Dibujar la línea de identidad (predicción ideal)
    min_val = min(min(y), min(predictions))
    max_val = max(max(y), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Ideal')
    plt.legend()
    
    # Guardar el gráfico en la carpeta results/
    output_path = os.path.join(base_path, 'results', 'pred_vs_true_mlp.png')
    plt.savefig(output_path)
    print(f"Gráfico guardado en {output_path}")

if __name__ == '__main__':
    evaluate_mlp_model()
