#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

def nothing(x):
    pass

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()
controlIn = pipeline.createXLinkIn()


xoutRgb.setStreamName("rgb")
controlIn.setStreamName('control')


# Properties
camRgb.setPreviewSize(1920, 1080)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
controlIn.out.link(camRgb.inputControl)
camRgb.preview.link(xoutRgb.input)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

currExp = 1
currISO = 100

def testChangeExp(x):
    global currExp, currISO
    currExp = x + 1
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(currExp, currISO)
    controlQueue.send(ctrl)
    print("Exp: ", currExp)

def testChangeIso(x):
    global currExp, currISO
    currISO = x + 100
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(currExp, currISO)
    controlQueue.send(ctrl)
    print("ISO: ", currISO)

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
# Exposure = [1, 33000]
cv2.createTrackbar('Exposure', 'Trackbars', 0, 32999, testChangeExp)
# ISO = [100, 1600]
cv2.createTrackbar('ISO', 'Trackbars', 0, 1500, testChangeIso)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    controlQueue = device.getInputQueue('control')
    testChangeExp(1000)
    testChangeIso(1600)

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
            # Flip the frame horizontally (Not required)
            # frame = cv2.flip( frame, 1 ) 
            
            # Convert the BGR image to HSV image.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get the new values of the trackbar in real time as the user changes 
            # them
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
            # Set the lower and upper HSV range according to the value selected
            # by the trackbar
            lower_range = np.array([l_h, l_s, l_v])
            upper_range = np.array([u_h, u_s, u_v])
            
            # Filter the image and get the binary mask, where white represents 
            # your target color
            mask = cv2.inRange(hsv, lower_range, upper_range)
        
            # You can also visualize the real part of the target color (Optional)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Converting the binary mask to 3 channel image, this is just so 
            # we can stack it with the others
            mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # stack the mask, orginal frame and the filtered result
            stacked = np.hstack((frame,res))

            # Show this stacked frame at 40% of the size.
            cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.8,fy=0.8))
            
            # If the user presses ESC then exit the program

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) #wait until any key is pressed

                    # If the user presses `s` then print this array.
            if key == ord('s'):
                
                thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
                print(thearray)
                
                # Also save this array as penval.npy
                # np.save('hsv_value',thearray)
                break

cv2.destroyAllWindows()