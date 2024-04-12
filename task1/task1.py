# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Size': [1400, 1600, 1700, 1875, 1100],
        'Price': [245000, 312000, 279000, 308000, 199000]}

# Creating a DataFrame
df = pd.DataFrame(data)

# Visualizing the data
plt.scatter(df['Size'], df['Price'])
plt.title('House Price vs. Size')
plt.xlabel('Size (sq. ft)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Fitting the linear regression model
X = df['Size'].values.reshape(-1, 1)
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Printing the coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Making predictions
size_new = np.array([[1500], [1800]])
predicted_prices = model.predict(size_new)
print("Predicted Prices:", predicted_prices)
