# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:28:52 2020

@author: oprad
"""

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("skin.h5")

@app.route('/')
def index():
    return render_template('UIndex.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            
            
            print("prediction",preds)
            
        index = ['Actinic Keratosis - Must undergo Cryotherapy.','Dermatofibroma - It is Harmless ,but need to be removed surgically.','Melanoma - It is a serious form of skin cancer,must be treated immediately.','Seborrheic Keratosis - It is a type of skin growth which is harmless.','Squamous Cell Carcinoma - It is a common type of skin cancer and can be treated by a Laser Surgery.']
        
        text = "The predicted Disease is " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    