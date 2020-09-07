import os
import imutils
import time
import tempfile
from flask import Flask, request, redirect, url_for, send_from_directory
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from PIL import Image

import PIL

import os
import glob


import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import os

#parser = argparse.ArgumentParser()
#parser.add_argument("--img1", type = str, required=True)
#parser.add_argument("--img2", type = str, required=True)
#args = parser.parse_args()
#tf.variable_scope()
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.8, 0.9]
factor = 0.709
margin = 0
input_image_size = 160
faces = []




#sess = tf.Session()
sess=tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.95:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    return -1
    
def getdetectedFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            print(face[4])
            if face[4] > 0.95:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_AREA)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]]})
    return faces
    
def saveFaces(img):
    savedetectedfaces='F:/computervision/facematch-master/facesdetected/'
    img = cv2.imread(img)
    h,w = img.shape[:2]
    print(h)
    print(w)
    if h > 800:
        img = imutils.resize(img,height=800)
    if w > 800:
        img = imutils.resize(img, width=800)
    faces = getdetectedFace(img)
    filename="face"
    ext='.jpg'
    i=0
    for face in faces:
        i+=1
        cv2.rectangle(img, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
        #cv2.imshow("each_face", face['face'])
        cv2.imwrite(savedetectedfaces+filename+str(i)+ext, face['face'])
        #cv2.waitKey(0)
        

    cv2.imshow("faces", img)
    #cv2.waitKey(0) waits till some key is pressed
    cv2.destroyAllWindows()
    
    

#UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS
    
def makeFaceList():
    basepath='F:/computervision/facematch-master/Ramesh/images/'
    for root, dirs, files in os.walk(basepath):
                    for filena in files:
                        print(filena)
                        img = cv2.imread(basepath+filena)
                        name=filena.split('.')[0]
        
                        img_size = np.asarray(img.shape)[0:2]
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        if not len(bounding_boxes) == 0:
                            for face in bounding_boxes:
                                if face[4] > 0.95:
                                    det = np.squeeze(face[0:4])
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                                    prewhitened = facenet.prewhiten(resized)
                                    faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened),'name':name})
    return faces    

faces1=makeFaceList()
@app.route('/im_size1', methods=['GET', 'POST'])
def upload_file1():
    distance=0
    faces2=[]
    start_time = time.time()
    empNames=[]
    img1=''
    if request.method == 'POST':
        file = request.files['name']
        if file and allowed_file(file.filename):
            print( '**found file', file.filename)
            tempfile.gettempdir()
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            im = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            picture= im.resize((240,240),Image.ANTIALIAS)
           
            picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename),optimize=True,quality=95)
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #img1 = cv2.imread(file.filename)
            '''saveFaces(os.path.join(app.config['UPLOAD_FOLDER'], filename))'''
            faces2=getFace(img1)
            #faces2=getFace(file.filename)
    
            #img2 = cv2.imread('F:/computervision/facematch-master/images/face1.jpg')
            #distance = compare2face(img1, img2)
            threshold = 1.10    # set yourself to meet your requirement
            #print("distance = "+str(distance))
            basepath='F:/computervision/facematch-master/Ramesh/images/'
            #print("Result = " + ("same person" if distance <= threshold else "not same person"))
            detectedfacefolder='F:/computervision/facematch-master/facesdetected/'
            #print(faces2)
            #print(faces)
            #print("Result = " + ("same person" if distance <= threshold else "not same person"))
            for i in range(len(faces2)):
                for j in range(len(faces)):
                    if faces2[i] and faces[j]:
                        # calculate Euclidean distance
                        #print('faces2'+faces2)
                        #print('faces'+faces)
                        distance = np.sqrt(np.sum(np.square(np.subtract(faces2[i]['embedding'], faces[j]['embedding']))))
                        #return dist
                    else:
                        distance=-1
                                               
                    if distance <= 0.88 and distance!= -1:
                                
                        empNames.append(faces[j]['name'])
                                
                                
                        break;
                            
                                
                    else:
                        Result= "not in list"
                        
            # for browser, add 'redirect' function on top of 'url_for'
            #return url_for('uploaded_file',filename=filename)
            #return jsonify({'msg': 'success', 'Result':Result })
            print ("Time taken :" + str(time.time() - start_time))
            for root2, dirs2, files2 in os.walk("F:/computervision/facematch-master/facesdetected/"):
                        for filena2 in files2:
                            os.remove(detectedfacefolder+filena2)

            return jsonify({'msg': 'success', 'Result': empNames})
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=name>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file_old(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
@app.route('/uploads',methods=['GET', 'POST'])
def uploaded_file():
    if request.method == 'POST':
        file = request.files['name']
        empname= request.form['empname']
        if file and allowed_file(file.filename):
            print( '**found file', file.filename,empname)
            #filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['BASE_FOLDER'], file.filename))
            file.save('F:/computervision/facematch-master/Ramesh/images/'+ empname+'.'+file.filename.split('.')[1])
            im = Image.open('F:/computervision/facematch-master/Ramesh/images/'+ empname+'.'+file.filename.split('.')[1])
            picture= im.resize((240,240),Image.ANTIALIAS)
           
            picture.save('F:/computervision/facematch-master/Ramesh/images/'+ empname+'.'+file.filename.split('.')[1],optimize=True,quality=95) 
            faces=[]
            makeFaceList()
            print(faces)
            
            return jsonify({'msg': 'Successfully '+empname+' Added'})

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=text name=empname>
          <input type=file name=name>
         <input type=submit value=Upload>
    </form>
    '''

'''if __name__ == '__main__':
	app.run(debug=True,host="192.168.0.101")'''
    
@app.route("/")
def hello():
    return "Hello world!"
if __name__ == "__main__":
    app.run(host="192.168.0.100",port="5000")
