import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import os
#%matplotlib inline

def display(img, cmap = "gray"):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap="gray")

im = cv2.imread("D:/img/pancard1.png")
plt.imshow(im)

import cv2
import matplotlib.pyplot as plt
from  PIL import Image
#%matplotlib inline

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
_,bin = cv2.threshold(gray,120,255,1) #light object on dark background
bin = cv2.dilate(bin, None)
bin = cv2.dilate(bin, None)
bin = cv2.erode(bin, None)
bin = cv2.dilate(bin, None)
contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

rc = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rc)
for p in box:
    pt= (int(p[0]),int(p[1]))
    print(pt)
   # cv2.circle(im,pt,5,(200,0,0),2)
display(im)

import cv2
#import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#%matplotlib inline

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

image = cv2.imread("D:/img/pancard1.png")
image_crop = Image.open("D:/img/pancard1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=[40, 60],
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("found {0} faces".format(len(faces)))

for (x,y,w,h) in faces:
    cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = im[y:y+h, x:x+w]

plt.imshow(image)
im_crop = image_crop.crop((x, y, (x+w), (y+h)))
#print(im_crop.shape)
im_crop.save('D:/img/faced.png')
img = cv2.imread("D:/img/faced.png",0)

#dimensions of image
dimensions=img.shape

#height and width of image
height= img.shape[0]
width=img.shape[1]
num = height**2 + width**2
hyt = num**0.5
hyt=int(hyt)
#clength="{:.2f}".format(hyt)
print("image dimensions:",dimensions)
print("height of the image:",height)
print("width of the image:",width)
print("length between two corner points:",hyt)

#img1 = imutils.resize(im_crop)
#img2 = img1[197:373,181:300]  #roi of the image
ans = []
for y in range(0, img.shape[0]):  #looping through each rows
     for x in range(0, img.shape[1]): #looping through each column
            #if img2[y, x] != 0:
                  ans = ans + [[x, y]]
ans = np.array(ans)
#for i in range(len(ans)):
    #print(ans[i])
print(ans)
#print(ans[-1])

plt.imshow(img)

plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
