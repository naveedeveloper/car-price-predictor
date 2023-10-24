from flask import Flask, render_template, request, redirect
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("cleaned_car_data.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route("/", methods=["GET"])
def index():
    companies = sorted(data['company'].unique())
    car_models = sorted(data['name'].unique())
    years = sorted(data['year'].unique())
    fuel_types = sorted(data['fuel_type'].unique())
    return render_template("index.html", companies = companies, car_models = car_models, years = years, fuel_types = fuel_types, show_price=False)

@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    kms_driven = request.form.get('kms_driven')
    print(company, car_model, year, fuel_type, kms_driven)
    prediction = model.predict(pd.DataFrame([[company, car_model, year, fuel_type, kms_driven]], columns=['company', 'name', 'year', 'fuel_type', 'kms_driven']))
    predicted_price = str(np.round(prediction[0], 2))
    return predicted_price


if __name__ == "__main__":
    app.run()