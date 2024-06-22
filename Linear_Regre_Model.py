import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

cakes = np.array([3, 5, 2, 4, 5, 8, 7, 9, 10, 11, 11, 13, 14, 15, 14, 17, 18, 17, 2, 21]).reshape(-1, 1)
flour_used = np.array([6.5, 10, 4, 9, 12.6, 16, 14.2, 18, 20, 24, 22.4, 26.5, 27, 30, 32.2, 34, 3, 38.4, 40, 42])

# Split data into training and testing sets
cakes_train, cakes_test, flour_train, flour_test = train_test_split(cakes, flour_used, test_size=0.2, random_state=42)

# Fit linear regression model on the training set
model = LinearRegression()
model.fit(cakes_train, flour_train)

# Predicted values for training and testing sets
predicted_flour_train = model.predict(cakes_train)
predicted_flour_test = model.predict(cakes_test)

print("cakes_test:",cakes_test)

# Calculate metrics for training set
mse_train = mean_squared_error(flour_train, predicted_flour_train)
mae_train = mean_absolute_error(flour_train, predicted_flour_train)
r2_train = r2_score(flour_train, predicted_flour_train)

# Calculate metrics for testing set
mse_test = mean_squared_error(flour_test, predicted_flour_test)
mae_test = mean_absolute_error(flour_test, predicted_flour_test)
r2_test = r2_score(flour_test, predicted_flour_test)

# Print metrics for training and testing sets
print("Training set metrics:")
print(f"Mean Squared Error (MSE): {mse_train:.2f}")
print(f"Mean Absolute Error (MAE): {mae_train:.2f}")
print(f"R-squared (R²): {r2_train:.2f}")

print("\nTesting set metrics:")
print(f"Mean Squared Error (MSE): {mse_test:.2f}")
print(f"Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"R-squared (R²): {r2_test:.2f}")

# Calculate residuals
residuals_train = flour_train - predicted_flour_train
residuals_test = flour_test - predicted_flour_test

# Plotting the training data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(cakes_train, flour_train, color='blue', label='Training data')
plt.scatter(cakes_test, flour_test, color='orange', label='Testing data')
plt.plot(cakes_train, predicted_flour_train, color='red', label='Regression line')

# Plotting residuals for training data
for i in range(len(cakes_train)):
    plt.vlines(cakes_train[i], predicted_flour_train[i], flour_train[i], color='green', linestyle='dashed')

# Plotting residuals for testing data
for i in range(len(cakes_test)):
    plt.vlines(cakes_test[i], predicted_flour_test[i], flour_test[i], color='purple', linestyle='dotted')

plt.title('Cakes vs Flour with Residuals')
plt.xlabel('Number of cakes')
plt.ylabel('Cups of flour')
plt.legend()
plt.grid(True)
plt.show()

print("Training residuals:", residuals_train)
print("Testing residuals:", residuals_test)

