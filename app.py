import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)

CORS(app)  

MODEL_PATH = os.path.join('models', 'optimized_xgboost_model.pkl')
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

DATA_PATH = os.path.join('data', 'raw_data.csv')

def load_data():
    data = pd.read_csv(DATA_PATH)
    
    countries = data['Area'].unique().tolist()
    crops = data['Item'].unique().tolist()
    years = data['Year'].unique().tolist()

    return countries, crops, years

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        country = data['country']
        crop = data['crop']
        year = data['year']
        averageRainfall = data['averageRainfall']
        pesticides = data['pesticides']
        avgTemp = data['avgTemp']

        countries, crops, years = load_data()

        country_map = {country: idx for idx, country in enumerate(countries)}
        crop_map = {crop: idx for idx, crop in enumerate(crops)}
        year_map = {year: idx for idx, year in enumerate(years)}

        if country not in country_map or crop not in crop_map:
            return jsonify({'error': 'País o cultivo no reconocido'}), 400

        if year not in year_map:
            year_map[year] = len(years)  
            years.append(year)  

        country_idx = country_map[country]
        crop_idx = crop_map[crop]
        year_idx = year_map[year]

        input_data = np.array([[country_idx, crop_idx, year_idx, averageRainfall, pesticides, avgTemp]])

        prediction = model.predict(input_data)

        result_kg = float(prediction[0]) * 0.1  

        return jsonify({'prediction': result_kg})

    except Exception as e:
        print(f"Error: {e}")  
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
