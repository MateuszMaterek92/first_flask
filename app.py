import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    przeznaczenie = request.form.get('przeznaczenie')
    wiek = request.form.get('wiek')
    kubatura = request.form.get('kubatura')

    if przeznaczenie == 'mieszkalny':
        przeznaczenie = 1
    else:
        przeznaczenie = 2

    integers = [przeznaczenie, wiek, kubatura]

    int_features = [int(x) for x in integers]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Przewidywane zuzycie ciepla wynosi {} GJ/rok'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)


