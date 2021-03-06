#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time
import math

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()
controlIn = pipeline.createXLinkIn()

controlIn.setStreamName('control')
xoutRgb.setStreamName("rgb")

# Properties
camRgb.setVideoSize(1920, 1080)
camRgb.setPreviewSize(1440, 810)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
controlIn.out.link(camRgb.inputControl)
camRgb.preview.link(xoutRgb.input)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

currExp = 1
currISO = 100


def testChangeExp(x):
    global currExp, currISO
    currExp = x + 1
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(currExp, currISO)
    controlQueue.send(ctrl)


def testChangeIso(x):
    global currExp, currISO
    currISO = x + 100
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(currExp, currISO)
    controlQueue.send(ctrl)


def testChangeF(x):
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(x)
    controlQueue.send(ctrl)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    controlQueue = device.getInputQueue('control')
    testChangeExp(1500)
    testChangeIso(1600)
    # testChangeF(8)

    first_iter = True
    backSub = cv2.createBackgroundSubtractorMOG2()
    points = []

    print(np.shape(qRgb))
    frame = None

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = inRgb.getCvFrame()

        # Retrieve 'bgr' (opencv format) frame

        if frame is not None:
            # cv2.imshow("rgb", frame)
            framed = frame  # frame[100:980, 200:1720]
            # print(frame.shape)

            # fgMask = backSub.apply(frame)
            # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            # # cv2.putText(frame, str(qRgb.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            # cv2.imshow('FG Mask', fgMask)
            # cut = cv2.bitwise_and(frame, frame, mask=fgMask)
            # cut_blur = cv2.medianBlur(cut, 5)
            # cv2.imshow("Cut", cut_blur)

            hsv = cv2.cvtColor(framed, cv2.COLOR_BGR2HSV)

            # define range of white color in HSV
            # change it according to your need !
            # lower_white = np.array([133, 0, 0], dtype=np.uint8)
            # upper_white = np.array([179, 255, 115], dtype=np.uint8)
            # lower_orange = np.array([5, 109, 187], dtype=np.uint8)
            # upper_orange = np.array([30, 255, 255], dtype=np.uint8)
            # lower_green = np.array([40, 55, 150], dtype=np.uint8)
            # upper_green = np.array([179, 160, 255], dtype=np.uint8)
            lower_green = np.array([59, 57, 102], dtype=np.uint8)
            upper_green = np.array([79, 255, 222], dtype=np.uint8)

            # Threshold the HSV image to get only white colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            # masko = cv2.inRange(hsv, lower_orange, upper_orange)
            # mask = cv2.bitwise_or(maskg, masko)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(framed, framed, mask=mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # sorting the contour based of area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            tracked = framed.copy()
            if contours:
                ((x, y), radius) = cv2.minEnclosingCircle(contours[0])
                # M = cv2.moments(contours[0])
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                center = (int(x), int(y), time.time())
                radius = int(radius)
                # only proceed if the radius meets a minimum size
                if radius > 6:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(tracked, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(tracked, (center[0], center[1]), 5, (0, 0, 255), -1)
                    points.append(center)
                # # if any contours are found we take the biggest contour and get bounding box
                # (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
                # # drawing a rectangle around the object with 15 as margin
                # if (5 < box_width < 1000 and 5 < box_height < 1000):
                #     points.append((int(x_min + (box_width/2)), int(y_min + (box_height/2))))
                #     cv2.circle(tracked, (points[-1][0], points[-1][1]), 20, (100, 150, 255), -1)
                #     # cv2.rectangle(framed, (x_min - 10, y_min - 10),(x_min + box_width + 10, y_min + box_height + 10),(0,255,0), 4)
            if first_iter:
                avg = np.float32(framed)
                first_iter = False

            # for i in range(1, len(points)):
            # tracked = framed.copy()
            if len(points) > 50:
                del points[0]

            for i in range(1, len(points)):
                # print(points[i])
                if (0 < points[i][0] < 1920 and 0 < points[i][1] < 1080):
                    cv2.circle(tracked, (points[i][0], points[i][1]), 2, (255, 0, 255), 2)
                    cv2.line(tracked, (points[i - 1][0], points[i - 1][1]), (points[i][0], points[i][1]), (0, 0, 100),
                             2)
            # cv2.accumulateWeighted(frame, avg, 0.005
            # result = cv2.convertScaleAbs(avg)
            # cv2.imshow('avg',result)

            # cv2.imshow('Mask',res)
            # cv2.imshow('mask',mask)

            new_frame_time = time.time()

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            # fps = (1 / (1+new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            try:
                if len(points) > 0:
                    vel = 0.006 * math.sqrt((points[len(points)-1][0] - points[len(points)-2][0])**2 + (points[len(points)-1][1] - points[len(points)-2][1])**2)/(points[len(points)-1][2] - points[len(points)-2][2])
                    cv2.putText(tracked, (str(round(vel, 4))+" ft/s"), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 3, cv2.LINE_AA)
            except ZeroDivisionError:
                vel = 0.000
                cv2.putText(tracked, ("0.000 ft/s"), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 3, cv2.LINE_AA)


            # converting the fps into integer
            # fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            # fps = str(fps)

            # putting the FPS count on the frame
            # cv2.putText(tracked, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 0), 2, cv2.LINE_AA)

            cv2.imshow('Tracking', tracked)

            # fps = qRgb.get(cv2.CAP_PROP_FPS)
            # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            key2 = cv2.waitKey(1)  # wait until any key is pressed
            print("Test")
        if key == ord('c'):
            points = []
cv2.destroyAllWindows()