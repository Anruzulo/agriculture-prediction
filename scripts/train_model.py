import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def load_data(X_path, y_path):
    """
    Carga los datos de características y etiquetas desde archivos CSV.
    """
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entrena el modelo XGBoost y evalúa su rendimiento.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror', 
        'max_depth': 6,  
        'eta': 0.1,     
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'eval_metric': 'rmse', 
    }

    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, 'test')], early_stopping_rounds=50)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo XGBoost con las métricas R2 y RMSE.
    """
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

def save_model(model, output_path):
    """
    Guarda el modelo entrenado en un archivo .pkl.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {output_path}")

if __name__ == "__main__":
    X_train_path = './data/splits/X_train.csv'
    y_train_path = './data/splits/y_train.csv'
    X_test_path = './data/splits/X_test.csv'
    y_test_path = './data/splits/y_test.csv'

    X_train, y_train = load_data(X_train_path, y_train_path)
    X_test, y_test = load_data(X_test_path, y_test_path)

    model = train_xgboost(X_train, y_train, X_test, y_test)

    evaluate_model(model, X_test, y_test)

    model_output_path = './models/xgboost_model.pkl'
    save_model(model, model_output_path)
