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