import numpy as np
import pandas as pd
import os

def generate_dataset(num_samples=1000, min_val=0.1, max_val=100.0):
    # Asegurarse de que la carpeta data exista en la raíz del proyecto
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    os.makedirs(base_path, exist_ok=True)
    
    # Generar valores aleatorios para a y b
    a_values = np.random.uniform(min_val, max_val, num_samples)
    b_values = np.random.uniform(min_val, max_val, num_samples)
    
    # Calcular la hipotenusa c usando la fórmula de Pitágoras
    c_values = np.sqrt(a_values**2 + b_values**2)
    
    # Crear un DataFrame con los datos
    data = {
        'a': a_values,
        'b': b_values,
        'c': c_values
    }
    df = pd.DataFrame(data)
    
    # Guardar el DataFrame en un archivo CSV dentro de la carpeta data/
    output_file = os.path.join(base_path, "triangle_dataset.csv")
    df.to_csv(output_file, index=False)
    print(f"Dataset guardado en {output_file}")

if __name__ == "__main__":
    generate_dataset()
