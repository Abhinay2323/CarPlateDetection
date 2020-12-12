# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract

CURRENT_DIRECTORY    = os.path.dirname( os.path.dirname( os.path.realpath( __file__ ) ) )
CURRENT_PACKAGE_NAME = os.path.basename( CURRENT_DIRECTORY ).rsplit('.', 1)[0]

UPLOAD_FOLDER = './flaskapp/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flaskapp/assets', 
            template_folder='./flaskapp')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('index.html')

@app.route('/upload_covidAndNormal.html')
def upload_covid():
   return render_template('upload_covid.html')

@app.route('/carplatedetection', methods = ['POST', 'GET'])
def uploaded_covid():
   if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   model1 = load_model('my_model_ocr_.h5')
   Xs=[]
   image = cv2.imread('./flaskapp/assets/images/upload_chest.jpg') # read file 
   image = cv2.resize(image,(224,224))
   #image = np.expand_dims(image, axis=0)
   Xs.append(np.array(image))
   Xs=np.array(Xs)
   Xs=Xs/255
   y_cnn = model1.predict(Xs)
   ny = y_cnn[0]*255
   print(ny.shape)
   imagesd = cv2.rectangle(image,(int(ny[0]),int(ny[1])),(int(ny[2]),int(ny[3])),(0, 255, 0))
   cv2.imwrite("output.png",imagesd)
   image1 = cv2.imread('output.png', 0)
   thresh1 = 255 - cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   print(thresh1.shape)
   ROI = thresh1[int(ny[3]):int(ny[1]),int(ny[2]):int(ny[0])]
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'   
   data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
   if data is None:
      res="No text identified"
   else:
      res=data
      
   return render_template('results_covid.html',result=res )


if __name__ == '__main__':
   app.secret_key = ".."
   app.run(debug=True)