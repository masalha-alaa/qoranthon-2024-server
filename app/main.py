import os.path

from flask import Flask, request, jsonify
from flask_cors import CORS
from app.torch_utils import ModelLoading
from yaml import safe_load


app = Flask(__name__)
CORS(app)
my_config = safe_load(open(os.path.join('app', 'config.yml')))
model_loading = ModelLoading(my_config['model']['filename'])


@app.route('/initialize', methods=['GET'])
def initialize():
    if request.method == 'GET':
        environment = my_config['environment']
        overwrite = my_config['model']['overwrite']
        clear_after_extract = my_config['model']['clear_after_extract']
        result = model_loading.initialize(overwrite=overwrite,
                                          environment=environment,
                                          clear_after_extract=clear_after_extract)
        response = jsonify({'result': result})
    else:
        response = jsonify({'result': 1})

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/encode', methods=['GET'])
def predict():
    if request.method == 'GET':
        sentence = request.args.get('sentence')
        prediction = model_loading.predict(sentence)
        data = {'sentence': sentence, 'encoding': prediction}
        response = jsonify(data)
    else:
        response = jsonify({'result': 1})

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
