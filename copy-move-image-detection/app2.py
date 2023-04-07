import sys
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
from flask import Flask,render_template,Response
# img='./'
def locateForgery(eps, min_sample, descriptors, key_points, image):
    l1=[]
    l2=[]
    clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors)
    size = np.unique(clusters.labels_).shape[0] - 1
    forgery = image.copy()
    # print(size)
    m=0
    if (size == 0) or (size==1) and (np.unique(clusters.labels_)[0] == -1):
        print('No Forgery Found!!')
        return None
    if size == 0:
    
        size = 1
    cluster_list = [[] for i in range(size)]
    for idx in range(len(key_points)):
        if clusters.labels_[idx] != -1:
            cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]), int(key_points[idx].pt[1])))
            # l1.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
            
            if m<size:
              l1.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
              m=m+1
              print(m)
            else:
              l2.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
           
    points=l1
    min_x1, min_y1 = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
    max_x1, max_y1 = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]
    # print(min_x,min_y,max_x,max_y)

 
    print(l1)
    print(l2)
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
      # print(l1[i])
      if i==0:
        center_coordinates = min_x2,min_y2
      elif(i==1):
        center_coordinates=max_x2,max_y2
      elif(i==2):
        center_coordinates=max_x2
      elif(i==3):
        center_coordinates=max_y2
# Radius of circle
    radius = 3
# Red color in BGR
    color = (0, 0, 255) 
# Line thickness of -1 px
    thickness = -1
    # forgery = cv2.circle(forgery, center_coordinates, radius, color, thickness)
    cv2.rectangle(forgery, (min_x1-10, min_y1-10), (max_x1+10, max_y1+10), (0, 0, 255), 2)
    cv2.rectangle(forgery, (min_x2-10, min_y2-10), (max_x2+10, max_y2+10), (0, 0, 255), 2)
    return forgery

def siftDetector(image):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # cv2.imshow('img3',gray)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points, descriptors

# image = cv2.imread('./DSC_0535tamp37.jpg')
# # cv2.imshow('img',image)
# key_points, descriptors = siftDetector(image)
# forgery = locateForgery(40, 2, descriptors, key_points, image)

# cv2.imshow('img1',image)
# cv2.imshow('img2',forgery)
# img = np.asarray(forgery)
# plt.imshow(img)
# while True: 
#     # _, frame = cap.read()

#     cv2.imshow('img1', forgery)
#     keyCode = cv2.waitKey(1)

#     if cv2.getWindowProperty('img1', cv2.WND_PROP_VISIBLE) <1:
#         break
# cv2.destroyAllWindows()

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/img')
def img():
    image = cv2.imread('./DSC_0535tamp37.jpg')
# cv2.imshow('img',image)
    key_points, descriptors = siftDetector(image)
    val=locateForgery(40, 2, descriptors, key_points, image)
    
    return Response(cv2.imshow('img1',val),mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__=="__main__":
    app.run(debug=True)