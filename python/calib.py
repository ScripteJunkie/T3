#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
# from matplotlib.pyplot import draw
import numpy as np
import sys
sys.path.append('../')
import envHandler

cal_img = []
p1 = []
mouse = 100, 100

pixel = 0, 0, 0

def bgr_to_hsv(b, g, r):
 
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
 
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
 
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
     
    # Isn't compatable with normal HSV, camera uses values to 255
    # to fix change 255 to 360, 85 to 120, and 170 to 240
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 255) % 255
 
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 85) % 255
 
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 170) % 255
 
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100 * 2.55
 
    # compute v
    v = cmax * 100 * 2.55
    return h, s, v

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

def mouseHandler(event, y, x, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        drawing = True
        p1.append([x, y])
        global pixel
        pixel = bgr_to_hsv(img1[x, y][0], img1[x, y][1], img1[x, y][2])
        print(pixel)
        # print('x = %d, y = %d'%(x, y))
        # print(1920-y,1080-x, y, x)
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
            in_rgb1 = q_rgb_list[0][0].tryGet()
            # in_rgb2 = q_rgb_list[1][0].tryGet()
            if in_rgb1 is not None:
                img1 = in_rgb1.getCvFrame()
                cv2.imshow("frame1", img1)
                # cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback('frame1', mouseHandler, img1)

            if cv2.waitKey(1) == ord('q'):
                if (pixel[0] != 0):
                    minm = np.clip([(pixel[0]-20), (pixel[1]-100), (pixel[2])-20], 0.0, 255.0).astype(int)
                    maxm = np.clip([(pixel[0]-10), 255, 255], 0.0, 255.0).astype(int)
                    print(minm, maxm)
                    envHandler.setVal("BALL_HSV_MIN", (str(minm[0]) + ", " + str(minm[1]) + ", " + str(minm[2])))
                    envHandler.setVal("BALL_HSV_MAX", (str(maxm[0]) + ", " + str(maxm[1]) + ", " + str(maxm[2])))
                break