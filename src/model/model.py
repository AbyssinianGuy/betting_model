import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the sports data (assuming it is in a CSV format)
data = pd.read_csv('./src/data/nba_data.csv')
cols_to_drop = ['game_id', 'commence_time', 'bookmaker',
                'last_update', 'in_play', 'description', 'point']
data = data.drop(cols_to_drop, axis=1)
# Create a new column called 'winning_team'
data['winning_team'] = np.where(
    data['home_team'] == data['label'], data['home_team'], data['away_team'])

# Drop the 'label' column
data = data.drop('label', axis=1)
# Convert the 'price' column to int64
data['price'] = data['price'].astype('int64')
print(data.head())

# Preprocess the data by encoding team names and scaling numerical features
le = LabelEncoder()
data['home_team'] = le.fit_transform(data['home_team'])
data['away_team'] = le.fit_transform(data['away_team'])
data['market'] = le.fit_transform(data['market'])
data['winning_team'] = le.fit_transform(data['winning_team'])

# Scale numerical features
numerical_features = ['home_team', 'away_team', 'price', 'market']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Convert input features to float32 data type
# data[numerical_features] = data[numerical_features].astype('float32')

# Split the data into training and testing sets
X = data.drop('winning_team', axis=1)
y = data['winning_team']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.dtypes)


# Create a neural network model using TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # The number of output nodes should match the number of classes (teams)
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Predict the winning team for new data
# new_data = np.array([[...]])  # Insert new data as a NumPy array
# predictions = model.predict(new_data)
# winning_team = le.inverse_transform([np.argmax(predictions[0])])
# print(f'Predicted winning team: {winning_team}')
