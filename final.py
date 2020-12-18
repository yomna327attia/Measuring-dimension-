# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as p
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#------------------------------------------------------------------------------

image = cv2.imread("len_8.8.jpg")

# Displaying the image
cv2.imshow("image",image)
# p.imshow(image)
# p.show()
# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# p.imshow(hsv)
# p.show()
#------------------------------------------------------------------------------
# define range of colors in HSV
#  blue color
lower_blue = np.array([90,50,50])
upper_blue = np.array([150,255,255])
# Red color
low_red = np.array([0, 150, 50])
high_red = np.array([10	,255,255])
low = [lower_blue,low_red]
upper=[upper_blue,high_red]
# Threshold the HSV image to get colors in order
i=0
for i in range(2)  :
       mask = cv2.inRange(hsv, low[i], upper[i])
       # p.imshow(mask)
       # p.show()
       # find contours
       contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       contours = [x for x in contours if cv2.contourArea(x) > 5000]
      # checking if one contour or more to consider the intersection case
       if len(contours) > 1:
           kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
           img = cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
           closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
           # p.imshow(closing)
           # p.show()
           contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           contours = [x for x in contours if cv2.contourArea(x) > 500]
       for x in contours:
           box = cv2.minAreaRect(x)
           box = cv2.boxPoints(box)
           box = np.array(box, dtype="int")
           box = perspective.order_points(box)
           (tl, tr, br, bl) = box
           cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 0), 2)
           if i ==0 :
               # getting the contour of max area
               areas = [cv2.contourArea(c) for c in contours]
               max_index = np.argmax(areas)
               # Reference object dimensions
               # Here for reference I have used a 5cm x 2cm
               ref_object = contours[max_index]
               box = cv2.minAreaRect(ref_object)
               box = cv2.boxPoints(box)
               box = np.array(box, dtype="int")
               box = perspective.order_points(box)
               (tl, tr, br, bl) = box
               dist_in_pixel = euclidean(tl, tr)
               dist_in_cm = 5
               pixel_per_cm = dist_in_pixel / dist_in_cm
               dist2_in_pixel = euclidean(br, bl)
               dist2_in_cm = 2
               pixel2_per_cm = dist2_in_pixel / dist2_in_cm

       ht = euclidean(tl, tr) / pixel_per_cm
       wid = euclidean(br, bl) / pixel2_per_cm
       print (ht,wid)
       (tltrX, tltrY) = midpoint(tl, tr)
       (blbrX, blbrY) = midpoint(bl, br)
       # compute the midpoint between the top-left and top-right points,
       # followed by the midpoint between the top-righ and bottom-right
       (tlblX, tlblY) = midpoint(tl, bl)
       (trbrX, trbrY) = midpoint(tr, br)
       # draw the object sizes on the image
       cv2.putText(image, "{:.1f}cm".format(ht), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 4,
                   (255, 255, 255), 3)
       cv2.putText(image, "{:.1f}cm".format(wid), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 4,
                   (255, 255, 255), 3)
       i=i+1
cv2.imshow("image",image)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
p.imshow(image)
p.show()

