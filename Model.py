# Model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from pickle import dump

data = pd.read_csv("laptop_data.csv")

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

features = data.drop('Price', axis=1)
pfeatures = PolynomialFeatures(degree=3)
nfeatures = pfeatures.fit_transform(features.values)  # Fit and transform
target = data['Price']

X_train, X_test, y_train, y_test = train_test_split(nfeatures, target, test_size=0.2, random_state=24)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    dump(model, f)

# Save the label encoders
with open("label_encoders.pkl", "wb") as f:
    dump(label_encoders, f)

# Save the PolynomialFeatures transformer
with open("pfeatures.pkl", "wb") as f:
    dump(pfeatures, f)
