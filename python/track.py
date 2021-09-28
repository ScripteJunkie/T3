import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

if __name__ == '__main__' :

    # Read image
    im = cv2.imread("/Users/ashtonmaze/Code/GitHub/T3/python/img/2.png", cv2.IMREAD_COLOR)

    # Select ROI
    # r = cv2.selectROI(im)

    # Crop image
    # imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    x1 = 1440
    x2 = 1600
    y1 = 600
    y2 = 715
    im = im[y1:y2, x1:x2]
    cap = im
    #cap = cv2.VideoCapture(‘ball_tracking.mp4’)
    redLower = np.array([0,10,170], dtype='uint8')
    redUpper = np.array([50,50,255], dtype='uint8')
    c = 0
    # frame_width = int(cap.get(3)) 
    # frame_height = int(cap.get(4)) 
    frame_width = im.shape[0]
    frame_height = im.shape[1]
    
    size = (frame_width, frame_height)
    #result = cv2.VideoWriter('balltracking.mp4’, cv2.VideoWriter_fourcc(*’MJPG’), 10, size) 
    #while True:
    #grapped, frame = cap.read()
    grapped = True
    frame = im
    if grapped == True:
        red = cv2.inRange(frame,redLower,redUpper)
        red = cv2.GaussianBlur(red,(3,3),0)
        cnts = cv2.findContours(red.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.circle(frame, (rect[0][0]+(rect[-1][0] - rect[0][0])//2,rect[1][1]+(rect[-1][-1]-rect[1][1])//2), 
        25, (0, 255, 0), -1)
        cv2.imshow("Ball Tracking", frame)
        #result.write(frame)
    # if cv2.waitKey() & 0xFF == ord("q"):
    #     break
    # else:
    #     break

    # Display cropped image
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow("Image", im)
    cv2.waitKey(0)