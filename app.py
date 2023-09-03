from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Function to categorize annual income
def categorize_income(income):
    if income < 30000:
        return 'low'
    elif 30000 <= income < 75000:
        return 'middle'
    else:
        return 'high'

# Load the trained model
model = joblib.load('src\model\knn_model.pkl')

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        frequent_flyer = request.form['frequent_flyer']

        annual_income = float(request.form['annual_income'])  # Convert to float
        income_category = categorize_income(annual_income) # Categorize annual income

        services_opted = int(request.form['services_opted'])
        social_media_synced = request.form['social_media_synced']
        booked_hotel = request.form['booked_hotel']

        # Encode categorical features (similar to how you did during EDA)
        frequent_flyer_encoded = 1 if frequent_flyer == 'Yes' else 0

        # Create a dictionary to map income categories to encoded values
        income_category_mapping = {
            'low': 1,
            'middle': 2,
            'high': 0
        }
        # Use the dictionary to encode the income category
        income_category_encoded = income_category_mapping.get(income_category, -1)  # Default to -1 if not found


        social_media_synced_encoded = 1 if social_media_synced == 'yes' else 0
        booked_hotel_encoded = 1 if booked_hotel == 'yes' else 0

        # Create a feature vector
        features = np.array([age, frequent_flyer_encoded, income_category_encoded,
                              services_opted, social_media_synced_encoded, booked_hotel_encoded]).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)[0]

        # Interpret the prediction
        if prediction == 0:
            result = 'Not Churn'
        else:
            result = 'Churn'

        return render_template('result.html', result=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
