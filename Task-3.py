from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === 1. Load Dataset & Preprocess ===
df = pd.read_csv(r"C:\Users\param\OneDrive\Documents\NareshIT\DataFiles\Visadataset.csv")

df.dropna(inplace=True)
df.drop("case_id", axis=1, inplace=True)

X = df.drop("case_status", axis=1)
y = df["case_status"]

cat_cols = ['continent', 'education_of_employee', 'has_job_experience',
            'requires_job_training', 'region_of_employment', 
            'unit_of_wage', 'full_time_position']

# Store individual encoders for each categorical column
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Encode target separately
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y.astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and metadata
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(label_encoder_y, open("label_encoder_y.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("feature_order.pkl", "wb"))

print("✅ Model, encoders, and column order saved.")

# === 2. Setup Flask App ===
app = Flask(__name__)

# Load model and metadata
model = pickle.load(open("model.pkl", "rb"))
label_encoder_y = pickle.load(open("label_encoder_y.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))

@app.route('/')
def home():
    return "✅ Flask app is running! Use POST /predict with correct feature list."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Received JSON:", data)

        # Validate feature count
        features = data.get("features")
        if not features:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        if len(features) != len(feature_order):
            return jsonify({
                "error": f"Expected {len(feature_order)} features, got {len(features)}"
            }), 400

        # Predict
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)
        predicted_class = label_encoder_y.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
