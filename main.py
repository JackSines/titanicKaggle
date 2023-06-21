import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare data
data = pd.read_csv('data/train.csv')

# Add some useful features
# Add a Title feature, taken from name
data['Title'] = data['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])

# Combine some of the less used titles (e.g., Lady) into Mr/Mrs
title_to_title_mapping = {
    'Sir': 'Mr',
    'Jonkheer': 'Mr',
    'Don': 'Mr',
    'Lady': 'Mrs',
    'Ms': 'Mrs',
    'Mme': 'Mrs',
    'Mlle': 'Mrs',
    'the Countess': 'Mrs'
}
data['Title'] = data['Title'].replace(title_to_title_mapping)
print(data['Title'].value_counts())
# Combine number of siblings + parents + 1 to get family size
data['Family_size'] = data['SibSp'] + data['Parch'] + 1

# Drop irrelevant columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Sex'], axis=1)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna('Q', inplace=True)

# Convert categorical variables to numerical
data['Embarked'] = data['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
title_to_int_mapping = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Major': 6, 'Col': 7, 'Capt': 8}
data['Title'] = data['Title'].map(title_to_int_mapping)

# Split features and labels
y = data['Survived']
X = data.drop('Survived', axis=1)

# Split into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardise features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Visual check of any empty values
print(data.isnull().sum())

# Set up model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],),
                          kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Validate
loss, accuracy = model.evaluate(x_val, y_val)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Load and prepare test data
test_data = pd.read_csv('data/test.csv')

# Isolate passenger ID's to create final csv later
passengerIds = test_data['PassengerId']

# Add some useful features
# Add a Title feature, taken from name
test_data['Title'] = test_data['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])
# Combine some of the less used titles (e.g., Lady) into Mr/Mrs
test_data['Title'] = test_data['Title'].replace(title_to_title_mapping)
print(test_data['Title'].value_counts())
# Combine number of siblings + parents + 1 to get family size
test_data['Family_size'] = test_data['SibSp'] + test_data['Parch'] + 1

# Drop irrelevant columns
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Sex'], axis=1)

# Fill missing values
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
data['Embarked'].fillna('Q', inplace=True)

# Convert categorical variables to numerical
test_data['Embarked'] = test_data['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
test_data['Title'] = test_data['Title'].map(title_to_int_mapping)

# Standardise
test_data_scaled = scaler.transform(test_data)

# Make predictions
predictions = model.predict(test_data_scaled)

# Convert probabilities to class labels
predictions_rounded = np.round(predictions).flatten().astype(int)

output = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions_rounded})
output.to_csv('predictions.csv', index=False)
