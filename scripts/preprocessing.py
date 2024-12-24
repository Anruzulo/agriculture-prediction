import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(input_path):
    """
    Carga los datos desde un archivo CSV.
    """
    return pd.read_csv(input_path)

def clean_data(df):
    """
    Realiza una limpieza básica de los datos:
    - Elimina filas con valores faltantes.
    - Convierte columnas categóricas y numéricas según sea necesario.
    - Elimina la columna 'Unnamed: 0' si está presente.
    """
    df = df.dropna()

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['Year'] = df['Year'].astype(int)
    df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(float)
    df['pesticides_tonnes'] = df['pesticides_tonnes'].astype(float)
    df['avg_temp'] = df['avg_temp'].astype(float)
    
    return df

def encode_categorical_features(df):
    """
    Codifica las columnas categóricas ('Area', 'Item') usando LabelEncoder.
    """
    encoders = {}
    for col in ['Area', 'Item']:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    input_path = "./data/raw_data.csv"
    output_processed_path = "./data/processed_data.csv"

    df = load_data(input_path)
    df_clean = clean_data(df)
    df_encoded, encoders = encode_categorical_features(df_clean)

    df_encoded.to_csv(output_processed_path, index=False)
    print(f"Datos procesados guardados en: {output_processed_path}")
