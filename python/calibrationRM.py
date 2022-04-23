import os
import time

import numpy as np
import cv2
import depthai as dai
import contextlib
import glob2 as glob
# first do: pip install future
# tkinter should work after this
import tkinter as tk

# Enable / Disable debug statements
verbose = True

currExp = 1
currISO = 100

fInit = 0
expInit = 0
isoInit = 0
wbInit = 0

fChange = None
expChange = None
isoChange = None
wbChange = None

dirPath = r'C:\Users\raman\PycharmProjects\T3\python\Output Images'
imgCount = len([
        f for f in os.listdir(dirPath)
        if f.endswith('.jpeg') and os.path.isfile(os.path.join(dirPath, f))
    ])

if verbose:
    print(f'Image Count = {imgCount}')


def testChangeF(x):
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(x)
    controlQueue.send(ctrl)


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


def testChangeWb(x):
    whiteb = x + 1000
    ctrl = dai.CameraControl()
    ctrl.setManualWhiteBalance(whiteb)
    controlQueue.send(ctrl)


def nothing(x):
    pass


cv2.namedWindow('Camera Controls')
cv2.resizeWindow('Camera Controls', 1900, 360)

# NOTE: cv2 trackbars cannot have minimum values other than zero. All max values shifted accordingly.

# Focus = [0, 255]
cv2.createTrackbar('Focus', 'Camera Controls', 0, 255, testChangeF)
# Exposure = [1, 33000]
cv2.createTrackbar('Exposure', "Camera Controls", 0, 32999, testChangeExp)
# ISO = [100, 1600]
cv2.createTrackbar('ISO', "Camera Controls", 0, 1500, testChangeIso)
# White Balance = [1000, 12000]
cv2.createTrackbar('White-Balance', "Camera Controls", 0, 11000, testChangeWb)


# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define sources
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    stillEncoder = pipeline.create(dai.node.VideoEncoder)

    # For the demo, just set a larger RGB preview size for OAK-D
    if device_type.startswith("OAK-D"):
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setVideoSize(1920, 1080)
        cam_rgb.setPreviewSize(1440, 810)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(cam_rgb.inputControl)

    stillMjpegOut = pipeline.create(dai.node.XLinkOut)
    stillMjpegOut.setStreamName('still')

    stillEncoder = pipeline.create(dai.node.VideoEncoder)
    stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    cam_rgb.still.link(stillEncoder.input)
    stillEncoder.bitstream.link(stillMjpegOut.input)

    # Create output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline


q_rgb_list = []
controlQueue_list = []

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
        if len(cameras) == 1:
            device_type = "OAK-1"
        elif len(cameras) == 3:
            device_type = "OAK-D"
        # If USB speed is UNKNOWN, assume it's a POE device
        if usb_speed == dai.UsbSpeed.UNKNOWN: device_type += "-POE"

        # Get a customized pipeline based on identified device type
        pipeline = getPipeline(device_type)
        print("   >>> Loading pipeline for:", device_type)
        device.startPipeline(pipeline)

        controlQueue = device.getInputQueue('control')

        stillQueue = device.getOutputQueue('still')

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxid + "-" + device_type
        # If reimplementing still frame capture, add stillQueue below
        q_rgb_list.append((controlQueue, stillQueue, q_rgb, stream_name))

    while True:
        # Adjust variables for cv2 trackbar shift (see note where trackbars defined)
        # focus = cv2.getTrackbarPos("Focus", "Camera Controls")
        # exposure = cv2.getTrackbarPos("Exposure", "Camera Controls") + 1
        # iso = cv2.getTrackbarPos("ISO", "Camera Controls") + 100
        # whiteb = cv2.getTrackbarPos("White-Balance", "Camera Controls") + 1000

        # If reimplementing still frame capture, add stillQueue
        for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:

            stillFrames = stillQueue.tryGetAll()
            for stillFrame in stillFrames:
                # Decode JPEG
                frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
                # Display
                cv2.imshow('still', frame)

            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                cv2.imshow(stream_name, in_rgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break

        elif cv2.waitKey(1) == ord('p'):
            cv2.imwrite(os.path.join(dirPath, f'img_{imgCount + 1}.jpeg'), frame)
            imgCount = len([
        f for f in os.listdir(dirPath)
        if f.endswith('.jpeg') and os.path.isfile(os.path.join(dirPath, f))
    ])
            if verbose:
                print(f'Image count = {imgCount}')
                print(f'saved image as img_{imgCount}.jpeg')

        elif cv2.waitKey(1) == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)