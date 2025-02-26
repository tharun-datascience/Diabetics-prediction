import numpy as np

import pickle


# Correct file path (ensure this file exists)
model_path = 'C:/Users/tharu/Downloads/Deep_learning/ML_DEPLOYMENTS/Diabetes/trained_model.sav'

# Load the trained model
loaded_model = pickle.load(open(model_path, 'rb'))

# Example input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape for a single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = loaded_model.predict(input_data_reshaped)

# Print the result
if prediction[0] == 0:
    print('The person is NOT diabetic')
else:
    print('The person is DIABETIC')
