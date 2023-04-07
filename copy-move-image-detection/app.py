from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import cv2
import sys
from sklearn.cluster import DBSCAN
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# This route renders the HTML page with the form to upload an image
@app.route('/')
def upload_form():
    return render_template('upload_form.html')
# This route is called when the form is submitted and the image is uploaded
@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    image = Image.open(filepath)
    image.save(filepath)
    image=np.uint8(image)
    print(image)
    # cv2.imshow('ase',image)
    # sift=cv2.SIFT_create() 
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # cv2.imshow('igray)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    l1=[]
    l2=[]
    # print(l1)
    # print(l2)
    # eps=40
    # min_samples=2
    clusters = DBSCAN(eps=40, min_samples=2).fit(descriptors)
    size = np.unique(clusters.labels_).shape[0] - 1
    forgery = image.copy()
    m=0
    if (size == 0) or (size==1) and (np.unique(clusters.labels_)[0] == -1):
        print('No Forgery Found!!')
        position=(10,50)
        cv2.putText(
     image, #numpy array on which text is written
     "No Forgery Found!", #text
     position, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1, #font size
     (209, 80, 0, 255), #font color
     3) #font s
        filename1='savedimage.jpg'
    
        filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        print(filename2)
        cv2.imwrite(filename2,image)
    # filename1='savedimage.jpg'
    # cv2.imwrite(filename1,forgery)
    # file.save(filename1)
    
    # file.save(filename2)
        return redirect(url_for('view_image', filename=filename1))
        # return None
    if size == 0:
        size = 1
    cluster_list = [[] for i in range(size)]
    for idx in range(len(key_points)):
        if clusters.labels_[idx] != -1:
            cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]), int(key_points[idx].pt[1]))) 
            if m<size:
              l1.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
              m=m+1
              print(m)
            else:
              l2.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
    points=l1
    min_x1, min_y1 = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
    max_x1, max_y1 = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]
    for i in range((2)):
      if i==0:
        center_coordinates = min_x1,min_y1
      elif(i==1):
        center_coordinates=max_x1,max_y1
      elif(i==2):
        center_coordinates=max_x1
      elif(i==3):
        center_coordinates=max_y1
    points=l2
    min_x2, min_y2 = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
    max_x2, max_y2 = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]
    for i in range((2)):
      if i==0:
        center_coordinates = min_x2,min_y2
      elif(i==1):
        center_coordinates=max_x2,max_y2
      elif(i==2):
        center_coordinates=max_x2
      elif(i==3):
        center_coordinates=max_y2
    radius = 3
    color = (0, 0, 255) 
    thickness = -1
    print(l1)
    print(l2)
    cv2.rectangle(forgery, (min_x1-10, min_y1-10), (max_x1+10, max_y1+10), (0, 0, 255), 2)
    cv2.rectangle(forgery, (min_x2-10, min_y2-10), (max_x2+10, max_y2+10), (0, 0, 255), 2)
    filename1='savedimage.jpg'
    
    filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    print(filename2)
    cv2.imwrite(filename2, cv2.cvtColor(forgery, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(filename2,forgery)
    # filename1='savedimage.jpg'
    # cv2.imwrite(filename1,forgery)
    # file.save(filename1)
    
    # file.save(filename2)
    return redirect(url_for('view_image', filename=filename1))
@app.route('/view/<filename>')
def view_image(filename):
    print('hi')
    return render_template('view_image.html',filename=filename)
if __name__ == '__main__':
    app.run(debug=True)
