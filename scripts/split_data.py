import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(input_path):
    """
    Carga los datos procesados desde un archivo CSV.
    """
    return pd.read_csv(input_path)

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en los datos.")
    
    X = df.drop(columns=[target_column])
    
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    input_path = "./data/processed_data.csv"
    
    target_column = "hg/ha_yield"
    
    df = load_data(input_path)
    
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    X_train.to_csv("./data/splits/X_train.csv", index=False)
    X_test.to_csv("./data/splits/X_test.csv", index=False)
    y_train.to_csv("./data/splits/y_train.csv", index=False)
    y_test.to_csv("./data/splits/y_test.csv", index=False)
    
    print("Datos divididos y guardados en la carpeta 'data/splits/'")