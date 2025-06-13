import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf # for neural networks
print(tf.__version__)

df = pd.read_csv("http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep='\t')
df = df[['Year Built','Gr Liv Area','Overall Qual','Neighborhood','Foundation','TotRms AbvGrd','Full Bath', 'Bedroom AbvGr','SalePrice']]
df.columns = ['Year Built','House Area (sqft)','Quality of Build','Neighborhood','Foundation Type','Total Rooms','Bathrooms','Bedrooms','SalePrice']

df.head()

df_categorical = df[['Foundation Type','Neighborhood']]

# One-hot encode categorical features
dummies = pd.get_dummies(df_categorical)
dummies.head()

df = df.drop(['Neighborhood','Foundation Type'], axis = 1)

# Append dummies
df = pd.concat([df, dummies], axis=1)

df.head()

y = df['SalePrice']

# normalize all the features (X)
X = df.drop(['SalePrice'], axis = 1)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Rejoin data
X = pd.DataFrame(X, columns = df.drop(['SalePrice'], axis = 1).columns) # convert back to pandas
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel("MSE")
  plt.legend(["Train MSE", "Test MSE"])

'''
plt.figure(figsize=(16, 6))
plot_graphs(history, 'loss')
'''

model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(1024, activation=tf.nn.relu),
     tf.keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dense(256, activation=tf.nn.relu),
     tf.keras.layers.Dense(128, activation=tf.nn.relu),
     tf.keras.layers.Dense(64, activation=tf.nn.relu),
     tf.keras.layers.Dense(32, activation=tf.nn.relu),
     tf.keras.layers.Dense(16, activation=tf.nn.relu),
     tf.keras.layers.Dense(1)
  ])

model.compile(optimizer = "adam", # don't worry about this, just use "adam"
              loss = 'mse')

history = model.fit(
    X_train, y_train, epochs=20,
    validation_data=(X_test, y_test)
)

# Define the preprocessing function
def preprocess_input(df_input, scaler, dummy_cols):
    df_cat = df_input[['Foundation Type', 'Neighborhood']]
    dummies_input = pd.get_dummies(df_cat)
    for col in dummy_cols:
        if col not in dummies_input:
            dummies_input[col] = 0
    dummies_input = dummies_input[dummy_cols]  # ensure correct order
    df_input = df_input.drop(['Foundation Type', 'Neighborhood'], axis=1)
    df_input = pd.concat([df_input, dummies_input], axis=1)
    df_input_scaled = scaler.transform(df_input)
    return df_input_scaled

# Define new house
new_house = pd.DataFrame([{
    'Year Built': 2005,
    'House Area (sqft)': 1800,
    'Quality of Build': 7,
    'Neighborhood': 'CollgCr',
    'Foundation Type': 'PConc',
    'Total Rooms': 6,
    'Bathrooms': 2,
    'Bedrooms': 3
}])

# Preprocess and predict
X_new = preprocess_input(new_house, scaler, dummies.columns)
predicted_price = model.predict(X_new)

print(f"Predicted Sale Price: ${predicted_price[0][0]:,.2f}")
