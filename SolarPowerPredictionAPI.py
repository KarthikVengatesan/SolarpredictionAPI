# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:20:55 2018

@author: 593378
"""

import numpy as np
import sklearn
import pickle
from sklearn.neural_network import MLPClassifier
from flask import Flask, request, jsonify

def predict_pv_output(num1,num2):
    
    output = {'output prediction of PV Solar':0}
    x_input = np.array([num1,num2]).reshape(1,2)
    filename = 'PV_model.pkl'
    ml = pickle.load(open(filename, 'rb'))
    output['output prediction of PV Solar'] = ml.predict(x_input)[0]
    #print output
    return output

app = Flask(__name__)

@app.route("/")
def index():
    return "PV Power Prediction"

@app.route("/pv_prediction", methods = ['GET'])
def calc_pv_Predict():
    body = request.get_data()
    header = request.headers
    
    try:
        num1 = int(request.args['x1'])
        num2 = int(request.args['x2'])
        if (num1 >= 0) and (num2 >= 0):
                res = predict_pv_output(num1,num2)
        else:
            res = {
                    'success': False,
                    'message': 'inputted data is not correct'}
    except:
        res = {
                'success': False,
                'message': 'Unknown error'
            }
    
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug = True, port = 8791)
