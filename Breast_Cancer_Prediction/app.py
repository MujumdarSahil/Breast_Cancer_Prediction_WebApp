from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        features = np.array([features])
        prediction = model.predict(features)
        if prediction == 1:
            prediction = "Malignant"
        else:
            prediction = "Benign"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
