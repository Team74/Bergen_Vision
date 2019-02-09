import cv2
import numpy as np
import math
import serial
import threading

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("BH", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("BS", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("BV", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("AH", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("AS", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("AV", "Trackbars", 255, 255, nothing)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    BH = cv2.getTrackbarPos("BH", "Trackbars")
    BS = cv2.getTrackbarPos("BS", "Trackbars")
    BV = cv2.getTrackbarPos("BV", "Trackbars")
    AH = cv2.getTrackbarPos("AH", "Trackbars")
    AS = cv2.getTrackbarPos("AS", "Trackbars")
    AV = cv2.getTrackbarPos("AV", "Trackbars")

    lower_green = np.array([BH, BS, BV])
    upper_green = np.array([AH, AS, AV])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Display the resulting frame
    cv2.imshow('basic', frame)
    cv2.imshow('mask', mask)

    # Stop the program from running
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
