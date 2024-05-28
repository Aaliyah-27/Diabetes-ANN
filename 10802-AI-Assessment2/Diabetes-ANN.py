import pandas as pd

# Task 1: Read the data
# Read the dataset from the specified CSV file path. Pandas' read_csv function
# is used to load data into a DataFrame, a powerful data structure for data analysis.
file_path = r'C:\\Users\\Aaliyah\\Documents\\Dataset of Diabetes .csv'
DataFrame = pd.read_csv(file_path)

# Preliminary data inspection
# Display the first few rows of the dataframe and summary statistics.
print(DataFrame.head())
print(DataFrame.info())
print(DataFrame.describe())

from sklearn.preprocessing import LabelEncoder

# Encode Gender (binary encoding)
# LabelEncoder is used to convert categorical labels into numeric form.
# This is necessary because machine learning models typically require numerical input.
label_encoder = LabelEncoder()
DataFrame['Gender'] = label_encoder.fit_transform(DataFrame['Gender'])

# Handle missing values
# Fill missing values with the mean of the column.
DataFrame.fillna(DataFrame.mean(), inplace=True)

from sklearn.preprocessing import OneHotEncoder

# One-hot encode CLASS
one_hot_encoder = OneHotEncoder(sparse_output=False)
class_encoded = one_hot_encoder.fit_transform(DataFrame[['CLASS']])
class_encoded_DataFrame = pd.DataFrame(class_encoded, columns=one_hot_encoder.categories_[0])

# Drop the original CLASS column and concatenate the new one-hot encoded columns
DataFrame = DataFrame.drop('CLASS', axis=1)
DataFrame = pd.concat([DataFrame, class_encoded_DataFrame], axis=1)

from sklearn.preprocessing import StandardScaler

# Standardize numerical features
numerical_features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
scaler = StandardScaler()
DataFrame[numerical_features] = scaler.fit_transform(DataFrame[numerical_features])

# Drop the ID and No_Pation columns as they are not useful for the model
DataFrame = DataFrame.drop(['ID', 'No_Pation'], axis=1)

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = DataFrame.drop(columns=['N', 'N ', 'P', 'Y', 'Y '])
y = DataFrame[['N', 'N ', 'P', 'Y']]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
