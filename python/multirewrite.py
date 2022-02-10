#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import time
import numpy as np
import multiprocessing

first_iter = True
backSub = cv2.createBackgroundSubtractorMOG2()
points1 = []
points2 = []
showp = True

def proc(frame, list):
    start = time.time()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([5, 109, 187], dtype=np.uint8)
    upper_white = np.array([30, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sorting the contour based of area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    tracked = frame.copy()
    if contours:
        #if any contours are found we take the biggest contour and get bounding box
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        #drawing a rectangle around the object with 15 as margin
        if (1 < box_width < 50 and 1 < box_height < 50):
            list.append((int(x_min + (box_width/2)), int(y_min + (box_height/2))))
            cv2.circle(tracked, (list[-1][0], list[-1][1]), int(max(box_height, box_width)/2), (100, 150, 255), -1)
            # cv2.rectangle(framed, (x_min - 10, y_min - 10),(x_min + box_width + 10, y_min + box_height + 10),(0,255,0), 4)
    # if first_iter:
    #     avg = np.float32(framed)
    #     first_iter = False

    # for i in range(1, len(points)):
    # tracked = framed.copy()

    for i in range(1, len(list)):
        # print(points[i])
        if (showp):
        # if (0 < list[i][0] < 1920 and 0 < list[i][1] < 1080):
            cv2.circle(tracked, (list[i][0], list[i][1]), 2, (255, 0, 255), 2)
            cv2.line(tracked, (list[i-1][0], list[i-1][1]), (list[i][0], list[i][1]), (0, 0, 100), 2)

    # cv2.imshow(stream_name, in_rgb.getCvFrame())
    # cv2.imshow(stream_name, tracked)
    end = time.time()
    fps = str(int((1/(end - start))+1))
    print(fps)
    return tracked

# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(720, 480)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(60)

    # Create output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

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
            out = proc(frame, points1)
            cv2.imshow("1", out)
        if in_rgb2 is not None:
            frame = in_rgb2.getCvFrame()
            out = proc(frame, points2)
            cv2.imshow("2", out)
        # for q_rgb, stream_name in q_rgb_list:
        #     in_rgb = q_rgb.tryGet()
        #     if in_rgb is not None:
        #         frame = in_rgb.getCvFrame()
        #         proc(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            showp = False if showp else True

        if key == ord('c'):
            points1=[]
            points2=[]