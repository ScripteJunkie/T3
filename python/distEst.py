#!/usr/bin/env python3

from turtle import position
import cv2
import depthai as dai
import contextlib
import time
import numpy as np
import math

# size = 6.125 # paddle (inches)
size = 1.57 # ball (inches)

# define range of white color in HSV
# change it according to your need !
# paddle
# lower_white = np.array([160, 144, 0], dtype=np.uint8)
# upper_white = np.array([179, 255, 255], dtype=np.uint8)
# ball
lower_white = np.array([0, 135, 135], dtype=np.uint8)
upper_white = np.array([40, 255, 255], dtype=np.uint8)

def proc(frame):
    start = time.time()
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    result = roi(frame)
    tracked = frame.copy()
    posDump = []
    cleaned = []
    prev = []
    if result:
        #if any contours are found we take the biggest contour and get bounding box
        # (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        (mask, x_min, y_min, box_width, box_height, contours) = result
        res = cv2.bitwise_and(frame,frame, mask= mask)
        #drawing a rectangle around the object with 15 as margin
        if (15 < box_width < 200 and 15 < box_height < 200):
            # Rough visual angle based on pixels/degree and dimension of the bounds
            vizAngZ = round(2*max(box_height, box_width)/(1920/68.7938), 2)
            # Distance estimate from the known dimensions and the objects visual angle
            disToCam = 4*(math.sin(math.radians(90-(vizAngZ/2)))*(size/2))/math.sin(math.radians(vizAngZ))
            # cv2.circle(tracked, (list[-1][0], list[-1][1]), int(max(box_height, box_width)/2), (100, 150, 255), -1)
            cv2.rectangle(tracked, (x_min - 0, y_min - 0),(x_min + box_width + 0, y_min + box_height + 0),(100,100,0), 1)
            cv2.putText(tracked, '6 1/8 in (0.155575 m)', (x_min - 0, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,12), 2)
            cv2.putText(tracked, str(vizAngZ) + " deg", (x_min - 0, y_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,12), 2)
            cv2.putText(tracked, str(round(disToCam, 3)) + " inch", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,100,12), 2)           
            posDump = [vizAngZ, disToCam, box_height*2, box_width*2, x_min, y_min]
        for i in contours:
            x, y, width, height = cv2.boundingRect(i)
            # for l in contours:
            #     if l <  or l >= j :
            #         res = False 
            #         break
            if 10000 > width*height > 100:# and 0.8*height < width < 1.2*height:
                cv2.rectangle(tracked, (x - 0, y - 0),(x + width + 0, y + height + 0),(0,255,0), 2)
                cv2.putText(tracked, str(width) + " " + str(height), (x - 0, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,12), 2)
        prev = contours
    return tracked, res, posDump

def roi(frameIn):
    frameIn = cv2.cvtColor(frameIn, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(frameIn, lower_white, upper_white)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sorting the contour based of area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if contours:
        # # rank = []
        # for i in contours:
        #     rect = cv2.minAreaRect(i)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     cv2.drawContours(frameIn,[box],0,(0,255, 255),2)
        # cv2.imshow("gngmgm", frameIn)
        # #     (x_min, y_min, box_width, box_height) = cv2.boundingRect(i)
        # #     crop = mask[(x_min-20):(x_min+(box_width+20)), (y_min-20):(y_min+(box_height+20))]
        # #     if np.sum(crop) > 0:
        # #         cv2.imshow("rank", crop)
        # #         density = np.sum(crop > 0)/np.sum(crop == 0)
        # #         rank.append(density)
        # #         print(np.argmax(rank))
        # #         (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[np.argmax(rank)])
        # #         return mask, x_min, y_min, box_width, box_height
        # #     else:
        # #         return mask, 0, 0, 0, 0
        # (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        # return mask, x_min, y_min, box_width, box_height
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        return mask, x_min, y_min, box_width, box_height, contours
    else: 
        return mask, 0, 0, 0, 0, []

# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(1920, 1080)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setPreviewKeepAspectRatio(True)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(60)

    # Create output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    cam_in = pipeline.createXLinkIn()
    cam_in.setStreamName("cam_in")
    cam_in.out.link(cam_rgb.inputControl)

    return pipeline

q_rgb_list = []

# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        print("=== Connected to " + device_info.getMxId())
        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()
        print("   >>> MXID:", mxid)
        print("   >>> Cameras:", *[c.name for c in cameras])
        print("   >>> USB speed:", usb_speed.name)

        device_type = "unknown"
        if   len(cameras) == 1: device_type = "OAK-1"
        elif len(cameras) == 3: device_type = "OAK-D"
        # If USB speed is UNKNOWN, assume it's a POE device
        if usb_speed == dai.UsbSpeed.UNKNOWN: device_type += "-POE"

        # Get a customized pipeline based on identified device type
        pipeline = getPipeline(device_type)
        print("   >>> Loading pipeline for:", device_type)
        device.startPipeline(pipeline)

        # Clear Queue
        device.getQueueEvents() #Might be in the wrong spot idk

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxid + "-" + device_type
        q_rgb_list.append((q_rgb, stream_name))

    while True:
        #this just loops through all unique frames per stream so is a waist of time
        # should be changed
        in_rgb1 = q_rgb_list[0][0].tryGet()
        in_rgb2 = q_rgb_list[1][0].tryGet()
        if in_rgb1 is not None:
            frame = in_rgb1.getCvFrame()
            out1, res1, pos1 = proc(frame)
        if in_rgb2 is not None:
            frame = in_rgb2.getCvFrame()
            out2, res2, pos2 = proc(frame)

        # posDump = [vizAngZ, disToCam, box_height*2, box_width*2, x_min, y_min]
        if len(pos1) > 0:
            try:
                angtoXaxis = round((abs(540-pos1[5]))/(1920/68.7938), 2)
                Zheight1 = math.sin(math.radians(90-(angtoXaxis/2)))*(size/2)/math.sin(math.radians(angtoXaxis/2))
                if (pos1[5] > 540):
                    Zheight1 *= -1
            except ZeroDivisionError:
                Zheight1 = 0
            # print("Camera 1 Z Estimate:" + str(round(Zheight1, 4)))
        if len(pos2) > 0:
            try:
                angtoXaxis = round((abs(540-pos2[5]))/(1920/68.7938), 2)
                Zheight2 = math.sin(math.radians(90-(angtoXaxis/2)))*(size/2)/math.sin(math.radians(angtoXaxis/2))
                if (pos2[5] > 540):
                    Zheight2 *= -1
            except ZeroDivisionError:
                Zheight2 = 0
            # print("Camera 2 Z Estimate:" + str(round(Zheight2, 4)))
        stackedTop = np.hstack((out1,out2))
        stackedBottom = np.hstack((res1,res2))
        Full = np.concatenate((stackedTop, stackedBottom), axis=0)
        final = cv2.resize(Full,None,fx=0.8,fy=0.8)
        cv2.imshow('Trackbars', final)
        # for q_rgb, stream_name in q_rgb_list:
        #     in_rgb = q_rgb.tryGet()
        #     if in_rgb is not None:
        #         frame = in_rgb.getCvFrame()
        #         proc(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # if key == ord('p'):
            # ctrl = dai.CalibrationHandler()
            # ctrl.setFov(dai.CameraBoardSocket.RGB, 20)
        if key == ord('c'):
            q_control = device.getInputQueue(name="cam_in")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            q_control.send(ctrl)
        if key == ord('a'):
            q_control = device.getInputQueue(name="cam_in")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            q_control.send(ctrl)
