import pandas as pd  # Import pandas library for data manipulation and analysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # Import necessary preprocessing tools from sklearn
from sklearn.model_selection import train_test_split  # Import function to split the data into training and testing sets
import tensorflow as tf  # Import TensorFlow library for building and training neural networks
from tensorflow.keras.models import Sequential  # Import Sequential model from Keras
from tensorflow.keras.layers import Dense, Dropout  # Import Dense and Dropout layers from Keras

# Task 1: Read the data
file_path = r'C:\\Users\\Aaliyah\\Documents\\Dataset of Diabetes .csv'  # Define the file path to the dataset (Update as required)
DataFrame = pd.read_csv(file_path)  # Read the dataset into a pandas DataFrame

# Preliminary data inspection
print(DataFrame.head())  # Display the first few rows of the DataFrame
print(DataFrame.info())  # Display summary information about the DataFrame
print(DataFrame.describe())  # Display statistical summary of the DataFrame

# Encode Gender (binary encoding)
label_encoder = LabelEncoder()  # Initialize the LabelEncoder
DataFrame['Gender'] = label_encoder.fit_transform(DataFrame['Gender'])  # Encode the 'Gender' column with numeric values

# Handle missing values: Fill missing values with the mean of the column, but only for numeric columns
numeric_columns = DataFrame.select_dtypes(include=['number']).columns  # Select only numeric columns
DataFrame[numeric_columns] = DataFrame[numeric_columns].fillna(DataFrame[numeric_columns].mean())  # Fill missing values with the mean for numeric columns

# One-hot encode CLASS
one_hot_encoder = OneHotEncoder(sparse_output=False)  # Initialize the OneHotEncoder with dense output
class_encoded = one_hot_encoder.fit_transform(DataFrame[['CLASS']])  # One-hot encode the 'CLASS' column
class_encoded_DataFrame = pd.DataFrame(class_encoded, columns=one_hot_encoder.categories_[0])  # Convert the encoded array to a DataFrame

# Drop the original CLASS column and concatenate the new one-hot encoded columns
DataFrame = DataFrame.drop('CLASS', axis=1)  # Drop the original 'CLASS' column
DataFrame = pd.concat([DataFrame, class_encoded_DataFrame], axis=1)  # Concatenate the one-hot encoded columns to the original DataFrame

# Standardize numerical features
numerical_features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']  # Define the list of numerical features
scaler = StandardScaler()  # Initialize the StandardScaler
DataFrame[numerical_features] = scaler.fit_transform(DataFrame[numerical_features])  # Standardize the numerical features

# Drop the ID and No_Pation columns as they are not useful for the model
DataFrame = DataFrame.drop(['ID', 'No_Pation'], axis=1)  # Drop the 'ID' and 'No_Pation' columns

# Define features (X) and target (y)
X = DataFrame.drop(columns=['N', 'N ', 'P', 'Y', 'Y '])  # Define the features by dropping the target columns
y = DataFrame[['N', 'N ', 'P', 'Y']]  # Define the target as the one-hot encoded 'CLASS' columns

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets

# Define the neural network model
model = Sequential([  # Initialize a Sequential model
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Add a dense layer with 64 units and ReLU activation
    Dropout(0.2),  # Add a dropout layer with 0.2 dropout rate
    Dense(32, activation='relu'),  # Add a dense layer with 32 units and ReLU activation
    Dropout(0.2),  # Add a dropout layer with 0.2 dropout rate
    Dense(y_train.shape[1], activation='softmax')  # Add an output layer with units equal to number of classes and softmax activation
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate with testing data
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))  
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from sklearn.metrics import confusion_matrix, classification_report  # Import metrics for model evaluation

# Predict the test set
y_pred = model.predict(X_test)  # Predict the output for the test set
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
y_true = np.argmax(y_test.values, axis=1)  # Convert one-hot encoded true labels to class labels

# Print unique values in y_true to verify actual classes
print("Unique values in y_true:", np.unique(y_true))  # Display unique class labels in the true labels

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)  # Generate confusion matrix
print("Confusion Matrix:")  # Print confusion matrix
print(cm)  # Display confusion matrix

# Generate the classification report
cr = classification_report(y_true, y_pred_classes, target_names=['N', 'N ', 'P'])  # Generate classification report
print("Classification Report:")  # Print classification report
print(cr)  # Display classification report

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])  # Plot training accuracy
plt.plot(history.history['val_accuracy'])  # Plot validation accuracy
plt.title('Model accuracy')  # Set plot title
plt.ylabel('Accuracy')  # Set y-axis label
plt.xlabel('Epoch')  # Set x-axis label
plt.legend(['Train', 'Test'], loc='upper left')  # Add legend
plt.show()  # Display plot

# Plot training & validation loss values
plt.plot(history.history['loss'])  # Plot training loss
plt.plot(history.history['val_loss'])  # Plot validation loss
plt.title('Model loss')  # Set plot title
plt.ylabel('Loss')  # Set y-axis label
plt.xlabel('Epoch')  # Set x-axis label
plt.legend(['Train', 'Test'], loc='upper left')  # Add legend
plt.show()  # Display plot
