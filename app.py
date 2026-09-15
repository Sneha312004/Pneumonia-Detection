import os

import numpy as np

from flask import Flask, render_template, request, redirect, url_for

from tensorflow.keras.models import load_model

from werkzeug.utils import secure_filename

from utils.preprocessing import preprocess

app = Flask(name)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'model/best_model2.keras'

model = load_model(MODEL_PATH)

def assign_tier(prob):

if prob <= 0.40:

    return "green", "Safe: Pneumonia very unlikely."

elif prob <= 0.80:

    return "yellow", "Caution: Possible pneumonia detected."

else:

    return "red", "Immediate Attention: High risk of pneumonia detected!"

@app.route('/', methods=['GET', 'POST'])

def index():

if request.method == 'POST':

    file = request.files['file']

    if file and file.filename:

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        x = preprocess(filepath)

        prob = float(model.predict(x)[0])

        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

        tier, message = assign_tier(prob)

        confidence = prob if label == "PNEUMONIA" else 1 - prob

        return render_template('index.html',

                               filename=filename,

                               prediction=label,

                               confidence=f"{confidence*100:.2f}%",

                               risk_tier=tier,

                               message=message)

return render_template('index.html')

@app.route('/display/<filename>')

def display_image(filename):

return redirect(url_for('static', filename=f'uploads/{filename}'))

if name == 'main':

app.run(debug=True)

this is the flask app created for the model