#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
from matplotlib.pyplot import draw
import numpy as np

captured, calibrated = False, False
cal_img = []
p1 = []
mouse = 100, 100

# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    # For the demo, just set a larger RGB preview size for OAK-D
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(1920, 1080)

    cam_rgb.setInterleaved(False)

    # Create output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline

drawing = False

def mouseHandler(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        drawing = True
        p1.append([x, y])
        # print('x = %d, y = %d'%(x, y))
        print(1920-x,1080-y)
    if event == cv2.EVENT_LBUTTONUP:
        # draw circle here (etc...)
        drawing = False
        # print('x = %d, y = %d'%(x, y))
        # p1.append([x, y])
        try:
            m = (y-p1[len(p1)-2][1])/(x-p1[len(p1)-2][0])
            y = (m*(x))-(m*p1[len(p1)-2][0])+(p1[len(p1)-2][1])
            cv2.line(img1, (p1[len(p1)-2][0], p1[len(p1)-2][1]), (int(x), int(y)), (0, 0, 255), 1)
            cv2.imshow("frame1", img1)
            # print(p1)
        except ZeroDivisionError:
            # print(x, y)
            return
    try:
        mouse = x, y
        # zoomed = zoom(img1, x, y, 5)[(5*y-100):(5*y+100), (5*x-100):(5*x+100)]
        # cv2.circle(zoomed, (100,100), radius=2, color=(0, 0, 255), thickness=5)
        # cv2.imshow("zoom", zoomed)
    except TypeError:
        return

def zoom(img, x, y, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

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

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxid + "-" + device_type
        q_rgb_list.append((q_rgb, stream_name))

    while True:
        if captured == True and calibrated == False:
            in_rgb1 = q_rgb_list[0][0].tryGet()
            in_rgb2 = q_rgb_list[1][0].tryGet()
            if in_rgb1 is not None:
                img1 = in_rgb1.getCvFrame()
                cv2.imshow("frame1", img1)
                # cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback('frame1', mouseHandler, img1)
                # zoomed = zoom(img1, mouse, 5)[(5*mouse[1]-100):(5*mouse[1]+100), (5*mouse[0]-100):(5*mouse[0]+100)]
                # cv2.circle(zoomed, (100,100), radius=2, color=(0, 0, 255), thickness=5)
                # cv2.imshow("zoom", zoomed)
            # if in_rgb2 is not None:
            #     img2 = in_rgb2.getCvFrame()
            #     cv2.imshow("2", img2)

            # img1 = cal_img[0]
            # img2 = cal_img[1]
            # cv2.imshow("frame1", img1)
            # cv2.imshow("frame2", img2)
            # cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
            # cv2.setMouseCallback('frame1', mouseHandler, img1)
            # calibrated = True

        # for q_rgb, stream_name in q_rgb_list:
        #     in_rgb = q_rgb.tryGet()
        #     if in_rgb is not None:
        #         if captured == False:
        #             time.sleep(3)
        #             cal_img.append(in_rgb.getCvFrame().copy())
        #         # cv2.imshow(stream_name, in_rgb.getCvFrame())
        captured = True

        if cv2.waitKey(1) == ord('q'):
            break