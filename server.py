from flask import Flask
app = Flask(__name__)
import cPickle as pickle
from flask import render_template, request
import json
from network import Network
import numpy as np

app.debug = True

weights = pickle.load(open('weights.p', 'rb'))
biases = pickle.load(open('biases.p', 'rb'))


net = Network([784, 30, 10])
net.weights = weights
net.biases = biases

def format_vector(img):
    for i in xrange(len(img)):
        img[i] = [img[i]]

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/img', methods=['POST'])
def process_img():
    img = request.get_json()
    img = np.reshape(np.array(img, dtype='float_'), (784,1))
    res = np.argmax(net.feedforward(img))
    return json.dumps({'digit':res}), 200, {'ContentType':'application/json'}
