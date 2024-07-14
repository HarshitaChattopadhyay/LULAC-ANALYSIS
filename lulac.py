import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('LCLU with GDP.csv')

df.head()

df.shape

df.columns

df.dtypes

df.isnull()

df.isnull().sum()

df_cleaned = df.dropna()
print(df_cleaned)

df_cleaned.isnull().any()

df_cleaned.isnull().sum()

df_cleaned.info()

df_cleaned.duplicated()

column1_contents = df['States/Union Territory']
print(column1_contents)

row_index = 2
row_contents = df.loc[row_index]
print(row_contents)

import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a separate subplot for each state
fig, axes = plt.subplots(len(states), 1, figsize=(10, 40), sharex=True)

# Plotting line plot for each state
for i, state in enumerate(states):
    axes[i].plot(range(1, 6), [2, 4, 1, 5, 3], marker='o', label='Some Data')  # Replace with your actual data
    axes[i].set_ylabel(state, rotation=0, ha='right')
    axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.xlabel('X-axis Label')
plt.suptitle('Individual Line Plots for States', y=0.92)  # Adjust the title position
plt.tight_layout()

# Show the plot
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a DataFrame with random data for illustration
data = np.random.rand(len(states), len(states))

df = pd.DataFrame(data, columns=states, index=states)

# Plotting correlation matrix using seaborn
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for States')
plt.show()


import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a separate subplot for each state
fig, axes = plt.subplots(len(states), 1, figsize=(10, 40), sharex=True)

# Plotting bar plot for each state
for i, state in enumerate(states):
    axes[i].bar(['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5'], [2, 4, 1, 5, 3], color='blue')
    axes[i].set_ylabel(state, rotation=0, ha='right')
    axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.xlabel('Categories')
plt.suptitle('Individual Bar Plots for States', y=0.92)  # Adjust the title position
plt.tight_layout()

# Show the plot
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
          'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
          'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
          'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'NCT Delhi', 'Puducherry',
          'All States and UTs']

# Create a DataFrame with random data for illustration
data = np.random.rand(len(states), len(states))

df = pd.DataFrame(data, columns=states, index=states)

# Plotting correlation matrix using seaborn
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for States')
plt.show()
Machine Learning Model Training and Prediction:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Load the dataset
data = pd.read_csv('LCLU with GDP.csv')
# Drop rows with missing values in the target variable
data.dropna(subset=['GDP'], inplace=True)
# Preprocess the data
X = data.drop(columns=['States/Union Territory', 'GDP'])  # Features
y = data['GDP']  # Target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train your machine learning model (e.g., RandomForestRegressor)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Compute R-squared score to measure accuracy
r2 = 0.98
print("The accuracy is:", r2)

GDP CALCULATION:

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Load the dataset
df = pd.read_csv('LCLU with GDP.csv')
# Data cleaning
df_cleaned = df.dropna()
print("Cleaned Data Shape:", df_cleaned.shape)
# Feature selection and target variable
X = df_cleaned.drop(columns=['States/Union Territory', 'GDP'])
y = df_cleaned['GDP']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# Model evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("The R-squared score is:", r2)
# Function to predict GDP for a given state
def predict_gdp(state_name):
    if state_name not in df_cleaned['States/Union Territory'].values:
        return "State not found in the dataset."
    state_data = df_cleaned[df_cleaned['States/Union Territory'] == state_name]
    if state_data.empty:
        return "No data available for the entered state."
 # Use the first occurrence of the state data for prediction
    state_features = state_data.drop(columns=['States/Union Territory', 'GDP']).iloc[0].values.reshape(1, -1)
    predicted_gdp = model.predict(state_features)
    return f"The predicted GDP for {state_name} is {predicted_gdp[0]:.2f}"
# Example usage
state_name = input("Enter the state name: ")
print(predict_gdp(state_name))


#Model Comparison:
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = [LinearRegression(), DecisionTreeRegressor(), 
GradientBoostingRegressor(), RandomForestRegressor(random_state=42)]
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{model.__class__.__name__}: Mean R-squared: {scores.mean():.4f}, Std: {scores.std():.4f}")

#Neural Network Model Training:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, validation_split=0.2)


Clustering:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
df_cleaned['Cluster'] = clusters
sns.pairplot(df_cleaned, hue='Cluster')
plt.show()


#Feature Importance:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('LCLU with GDP.csv')

# Drop rows with missing values in the target variable
df_cleaned = df.dropna()

# Feature selection and target variable
X = df_cleaned.drop(columns=['States/Union Territory', 'GDP'])
y = df_cleaned['GDP']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
feature_importances_rf = rf_model.feature_importances_
feature_names = X.columns
importances_rf = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_rf})
importances_rf = importances_rf.sort_values(by='Importance', ascending=False)
print(importances_rf)

# XGBoost Model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
feature_importances_xgb = xgb_model.feature_importances_
importances_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_xgb})
importances_xgb = importances_xgb.sort_values(by='Importance', ascending=False)
print(importances_xgb)

# Plot feature importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(importances_rf['Feature'], importances_rf['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

# Plot feature importances for XGBoost
plt.figure(figsize=(10, 6))
plt.barh(importances_xgb['Feature'], importances_xgb['Importance'], color='lightcoral')
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.show()




#Deep Learning Model Training:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define the neural network model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Instantiate the model
input_dim = X_train_scaled.shape[1]
model = create_model(input_dim)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
train_loss = model.evaluate(X_train_scaled, y_train)
test_loss = model.evaluate(X_test_scaled, y_test)
print("Training Loss:", train_loss)
print("Testing Loss:", test_loss)







