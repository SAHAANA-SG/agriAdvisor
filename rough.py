import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset into a DataFrame
data = pd.read_csv('Crop_recommendation.csv')

# Extract features (X) and target variable (y)
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the StandardScaler to your training data
scaler_standard = StandardScaler()
scaler_standard.fit(X_train)

# Save the fitted StandardScaler to a file named 'standscaler.pkl'
with open('standscaler.pkl', 'wb') as file:
    pickle.dump(scaler_standard, file)

# Fit the MinMaxScaler to your training data
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)

# Save the fitted MinMaxScaler to a file named 'minmaxscaler.pkl'
with open('minmaxscaler.pkl', 'wb') as file:
    pickle.dump(scaler_minmax, file)

# Train the Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Save the trained model to a file named 'model.pkl'
with open('model.pkl', 'wb') as file:
    pickle.dump(rfc, file)
