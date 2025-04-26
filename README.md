# -END-TO-END-ATA-SCIENCE-PROJECT

Company: CODTECH IT SOLUTION

NAME : PARAMESH BABU BANDA

INTERN ID : CT04WS27

DOMAIN : DATA SCIENCE

MENTOR : NEELA SANTHOSH KUMAR

Step 1: Data Preparation and Model Building
Import Libraries
Necessary libraries like pandas, numpy, scikit-learn, pickle, and flask are imported.

Load and Clean the Dataset

The visa application dataset (Visadataset.csv) is loaded.

Missing values are removed using dropna().

The case_id column (not useful for prediction) is dropped.

Separate Features and Target

X: input features (like education, job experience, etc.).

y: the target variable (case_status) which needs to be predicted.

Encode Categorical Features

Several columns like 'continent', 'education_of_employee', and 'unit_of_wage' are categorical.

These are encoded into numbers using LabelEncoder.

Each column's encoder is saved separately to ensure consistency later during prediction.

Encode the Target Column

The target case_status is also label-encoded to convert categories (like "Certified", "Denied") into numbers.

Train-Test Split

The data is split into 80% training and 20% testing using train_test_split().

Train the Model

A Random Forest Classifier is trained on the training data.

Save the Model and Encoders
Using pickle, the following are saved:

Trained model (model.pkl)

Target label encoder (label_encoder_y.pkl)

Encoders for categorical features (encoders.pkl)

Feature column order (feature_order.pkl)

✅ A message is printed to confirm saving.

Step 2: Flask API Development
Initialize Flask App
A simple Flask app is created.

Load Saved Files

Model, target label encoder, and feature order are loaded back into the app.

Define API Routes

/ Route:
Returns a welcome message that says the API is running.

/predict Route (POST method):

Accepts JSON input containing a "features" key.

Validates whether the correct number of features is received.

Converts the features into a NumPy array.

Makes a prediction using the loaded model.

Converts the predicted number back to the original visa status using the label encoder.

Returns the prediction as a JSON response.

If any error occurs, it sends back a proper error message.

Run the Server
Finally, app.run(debug=True) starts the Flask server locally, allowing users to send prediction requests.

Conclusion
This code builds a full machine learning deployment system:

Preprocessing real-world data

Training and saving a machine learning model

Developing a web service to handle real-time predictions

It connects Data Science with Web Development, showing a practical way to deploy ML models in production.

![Image](https://github.com/user-attachments/assets/04f81519-d3c6-4f21-b602-6ebd2b855ddb)

