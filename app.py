from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from flask import Flask,render_template,request,url_for
app =Flask(__name__)
model = load_model('crop.h5')
def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)
@app.route('/')
def hello_world():
    return render_template('demo.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    
    Classes = ["Potato___Early_blight","Potato___Late_blight","Potato___healthy"]
    result = model.predict_classes([prepare('C:/Users/vaibavalaxmi/Downloads/archive/test_set/Potato___Late_blight/2.JPG')])
    disease=image.load_img('C:/Users/vaibavalaxmi/Downloads/archive/test_set/Potato___Late_blight/2.JPG')
    
    val=Classes[int(result)]
    return render_template('demo.html',pred='The analysis says {}'.format(val))
if __name__== '__main__':
    app.run(debug=True)