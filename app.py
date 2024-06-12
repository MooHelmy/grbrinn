from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    return image_arr

classes = ['Normal','Stroke']
model=load_model("modelMobilenet.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Brain Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        prediction = result[0][0]
        if prediction<.5:
            predictionlablle=classes[0]
        else:
            predictionlablle=classes[1]
        data={
            "predicted value":result[0][0],
             "predicted":predictionlablle
            }
        print(prediction)
        return jsonify({'prediction': predictionlablle})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        prediction = result[0][0]
         
        print(prediction)
        if prediction<.5:
            predictionlablle=classes[0]
        else:
            predictionlablle=classes[1]

            
            

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Brain Image Classification")


if __name__ == '__main__':
    app.run(debug=True, port=5000)