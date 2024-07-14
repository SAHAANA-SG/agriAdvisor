from flask import Flask, request, render_template
app = Flask(__name__, static_url_path='/static')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from CSV file
df = pd.read_csv('Crop_recommendation.csv')  # Replace 'Crop_recommendation.csv' with the actual filename

# Separate features (X) and target (y)
X = df.drop(columns=['label'])
y = df['label']

# Scale the input parameters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Function to recommend crop using Random Forest Classifier
def recommendation(N, P, K, temperature, humidity, pH, rainfall, X_scaled, y):
    # Scale the input parameters
    features = [[N, P, K, temperature, humidity, pH, rainfall]]
    features_scaled = scaler.transform(features)

    # Create and fit Random Forest model
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_scaled, y)

    # Use model to predict
    prediction = rfc.predict(features_scaled)
    
    # Print the prediction for debugging
    print("Predicted crop:", prediction[0])

    return prediction[0]

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Make recommendation
        recommended_crop = recommendation(N, P, K, temp, humidity, pH, rainfall, X_scaled, y)
        return recommended_crop

if __name__ == "__main__":
    app.run(debug=True)