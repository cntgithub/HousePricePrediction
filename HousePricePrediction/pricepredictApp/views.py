from django.shortcuts import render
import numpy as np
import joblib

# Load the model and both scalers
model = joblib.load(r'C:\Users\USER\Desktop\house price prediction\HousePricePrediction\pricepredictApp\linear_model.pkl')
scaler_X = joblib.load(r'C:\Users\USER\Desktop\house price prediction\HousePricePrediction\pricepredictApp\scaler_X.pkl')  # For input features
scaler_y = joblib.load(r'C:\Users\USER\Desktop\house price prediction\HousePricePrediction\pricepredictApp\scaler_y.pkl')  # For target variable

def homepage(request):
    return render(request, 'index.html')

def predict_price(request):
    # Get data from form submission
    bedrooms = int(request.GET['bedrooms'])
    area = int(request.GET['area'])
    bathrooms = int(request.GET['bathrooms'])
    stories = int(request.GET['stories'])
    mainroad = 1 if request.GET['mainroad'] == 'yes' else 0
    guestroom = 1 if request.GET['guestroom'] == 'yes' else 0
    basement = 1 if request.GET['basement'] == 'yes' else 0
    hotwaterheating = 1 if request.GET['hotwaterheating'] == 'yes' else 0
    airconditioning = 1 if request.GET['airconditioning'] == 'yes' else 0
    parking = int(request.GET['parking'])
    prefarea = 1 if request.GET['prefarea'] == 'yes' else 0
    furnishingstatus = 0 if request.GET['furnishingstatus'] == 'unfurnished' else (1 if request.GET['furnishingstatus'] == 'semi-furnished' else 2)

    # Prepare input data
    input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                            airconditioning, parking, prefarea, furnishingstatus]])

    # Scale input features
    input_scaled = scaler_X.transform(input_data)

    # Predict the price
    predicted_scaled_price = model.predict(input_scaled)

    # Inverse transform the predicted price to actual price
    actual_price = scaler_y.inverse_transform(predicted_scaled_price.reshape(-1, 1))

    return render(request, 'price.html', {'predicted_price': round(actual_price[0][0], 2)})
