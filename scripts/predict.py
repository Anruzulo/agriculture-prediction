import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def load_data(X_path):
    """
    Carga los datos de características desde un archivo CSV (para hacer predicciones).
    """
    return pd.read_csv(X_path)

def load_model(model_path):
    """
    Carga el modelo entrenado desde un archivo .pkl.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, X):
    """
    Realiza las predicciones utilizando el modelo cargado.
    """
    return model.predict(X)

def evaluate_predictions(y_true, y_pred):
    """
    Evalúa las predicciones utilizando R2 y RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

def save_predictions(y_pred, output_path):
    """
    Guarda las predicciones en un archivo CSV.
    """
    pred_df = pd.DataFrame(y_pred, columns=['Predictions'])
    pred_df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")

if __name__ == "__main__":
    X_test_path = './data/splits/X_test.csv'  
    model_path = './models/optimized_xgboost_model.pkl'  
    output_predictions_path = './predictions/predictions.csv' 

    X_test = load_data(X_test_path)
    model = load_model(model_path)

    y_pred = make_predictions(model, X_test)

    y_test_path = './data/splits/y_test.csv'
    y_test = pd.read_csv(y_test_path)
    evaluate_predictions(y_test, y_pred)

    save_predictions(y_pred, output_predictions_path)