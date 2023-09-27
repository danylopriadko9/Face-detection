import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 400)
cap.set(4, 300)

while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier('faces.xml')

    result = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in result:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    cv2.imshow('Result', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# img = cv2.imread('images/PRI_223554170.webp')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = cv2.CascadeClassifier('faces.xml')
#
# results = faces.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
#
# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
#
# cv2.imshow('Result', img)
# cv2.waitKey(0)