import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

data = pd.read_excel('house_data.xlsx')

data_encoded = pd.get_dummies(data, columns=['neighborhood', 'condition', 'amenities'])

X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('House Price Prediction (Scatter Plot)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
indices = np.arange(len(y_test))
bar_width = 0.35
opacity = 0.8

plt.bar(indices, y_test, bar_width, alpha=opacity, color='b', label='Actual Price')
plt.bar(indices + bar_width, y_pred, bar_width, alpha=opacity, color='r', label='Predicted Price')

plt.xlabel('House Index')
plt.ylabel('Price')
plt.title('House Price Prediction (Bar Graph)')
plt.xticks(indices + bar_width, indices)
plt.legend()

plt.tight_layout()
plt.show()
