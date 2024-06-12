import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import tensorflow as tf
from flask import Flask, request, jsonify

# Initialize Firebase app (replace with your credentials)
cred = credentials.Certificate('scan-care-94bee-firebase-adminsdk-pac7n-7b3f66687b.json')  # Path to your service account key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load your TensorFlow model
model = tf.keras.models.load_model('mobilenet.tflite')

# Define a Flask app to handle API requests
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Preprocess the input data (replace with your preprocessing logic)
    image = data['image']  # Assuming your input is an image
    # ... (your preprocessing steps)
    processed_image = tf.convert_to_tensor(image)

    # Make prediction using your model
    prediction = model.predict(processed_image)
    predicted_class = tf.argmax(prediction).numpy()

    # Save the prediction to Firestore (optional)
    doc_ref = db.collection('predictions').document()
    doc_ref.set({'image': image, 'prediction': predicted_class})

    # Return the prediction result
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all interfaces (0.0.0.0)
