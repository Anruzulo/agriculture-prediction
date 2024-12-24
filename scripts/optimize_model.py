import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def load_data(X_path, y_path):
    """
    Carga los datos de características y etiquetas desde archivos CSV.
    """
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y

def optimize_xgboost(X_train, y_train):
    """
    Realiza la optimización de los hiperparámetros usando RandomizedSearchCV.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

    param_dist = {
        'max_depth': [3, 5, 6, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  
        scoring='neg_mean_squared_error',  
        cv=5,  
        verbose=2,  
        random_state=42,
        n_jobs=-1  
    )

    randomized_search.fit(X_train, y_train)

    print("Mejores parámetros encontrados:")
    print(randomized_search.best_params_)
    
    return randomized_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo optimizado con las métricas R2 y RMSE.
    """
    y_pred = model.predict(X_test)

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
    print(f"Modelo optimizado guardado en: {output_path}")

if __name__ == "__main__":
    X_train_path = './data/splits/X_train.csv'
    y_train_path = './data/splits/y_train.csv'
    X_test_path = './data/splits/X_test.csv'
    y_test_path = './data/splits/y_test.csv'

    X_train, y_train = load_data(X_train_path, y_train_path)
    X_test, y_test = load_data(X_test_path, y_test_path)

    optimized_model = optimize_xgboost(X_train, y_train)

    evaluate_model(optimized_model, X_test, y_test)

    model_output_path = './models/optimized_xgboost_model.pkl'
    save_model(optimized_model, model_output_path)