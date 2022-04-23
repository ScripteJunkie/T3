import os
import time

import numpy as np
import cv2
import depthai as dai
import contextlib
import glob2 as glob
import threading
# first do: pip install future
# tkinter should work after this
import tkinter as tk

# Enable / Disable debug statements
verbose = True
doWhileLoop = True

# currExp = 1
# currISO = 100
#
# fInit = 0
# expInit = 0
# isoInit = 0
# wbInit = 0
#
# fChange = None
# expChange = None
# isoChange = None
# wbChange = None
#
# dirPath = r'C:\Users\raman\PycharmProjects\T3\python\Output Images'
# imgCount = len([
#     f for f in os.listdir(dirPath)
#     if f.endswith('.jpeg') and os.path.isfile(os.path.join(dirPath, f))
# ])
#
# if verbose:
#     print(f'Image Count = {imgCount}')
#
#
# def testChangeF(x):
#     for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:
#         ctrl = dai.CameraControl()
#         ctrl.setManualFocus(x)
#         controlQueue.send(ctrl)
#
#
# def testChangeExp(x):
#     for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:
#         global currExp, currISO
#         currExp = x + 1
#         ctrl = dai.CameraControl()
#         ctrl.setManualExposure(currExp, currISO)
#         controlQueue.send(ctrl)
#
#
# def testChangeIso(x):
#     global currExp, currISO
#     for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:
#         currISO = x + 100
#         ctrl = dai.CameraControl()
#         ctrl.setManualExposure(currExp, currISO)
#         controlQueue.send(ctrl)
#
#
# def testChangeWb(x):
#     for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:
#         whiteb = x + 1000
#         ctrl = dai.CameraControl()
#         ctrl.setManualWhiteBalance(whiteb)
#         controlQueue.send(ctrl)
#
#
# def nothing(x):
#     pass
#
#
# cv2.namedWindow('Camera Controls')
# cv2.resizeWindow('Camera Controls', 1900, 360)
#
# # NOTE: cv2 trackbars cannot have minimum values other than zero. All max values shifted accordingly.
#
# # Focus = [0, 255]
# cv2.createTrackbar('Focus', 'Camera Controls', 0, 255, testChangeF)
# # Exposure = [1, 33000]
# cv2.createTrackbar('Exposure', 'Camera Controls', 0, 32999, testChangeExp)
# # ISO = [100, 1600]
# cv2.createTrackbar('ISO', 'Camera Controls', 0, 1500, testChangeIso)
# # White Balance = [1000, 12000]
# cv2.createTrackbar('White-Balance', 'Camera Controls', 0, 11000, testChangeWb)


class cameraOAK:

    # ISSUE BELOW: VARIABLE DEVICE NEVER USED SO IT IS ONLY CREATING TWO INSTANCES OF THE SAME CAMERA,
    # TOO TIRED TO FIX, FIX FIRST THING TOMORROW!

    def __init__(self, device):
        global doWhileLoop

        self.device = device

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()

        # Define sources
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)

        self.xout_rgb.setStreamName("rgb")

        # Define source sizes
        self.cam_rgb.setVideoSize(1920, 1080)
        self.cam_rgb.setPreviewSize(1440, 810)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # self.controlIn = self.pipeline.create(dai.node.XLinkIn)
        # self.controlIn.setStreamName("control")
        # self.controlIn.out.link(self.cam_rgb.inputControl)
        #
        # self.stillMjpegOut = self.pipeline.create(dai.node.XLinkOut)
        # self.stillMjpegOut.setStreamName('still')
        #
        # self.stillEncoder = self.pipeline.create(dai.node.VideoEncoder)
        # self.stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        # self.cam_rgb.still.link(self.stillEncoder.input)
        # self.stillEncoder.bitstream.link(self.stillMjpegOut.input)

        # Create output
        self.cam_rgb.preview.link(self.xout_rgb.input)

        self.mxid = None
        self.cameras = None
        self.usb_speed = None

        if verbose:
            print("=== Connected to " + dai.Device(self.pipeline).getMxId())
        self.mxid = dai.Device(self.pipeline).getMxId()
        self.cameras = dai.Device(self.pipeline).getConnectedCameras()
        self.usb_speed = dai.Device(self.pipeline).getUsbSpeed()
        if verbose:
            print("   >>> MXID:", self.mxid)
            print("   >>> Cameras:", *[c.name for c in self.cameras])
            print("   >>> USB speed:", self.usb_speed.name)

        self.controlQueue = None
        self.stillQueue = None
        self.q_rgb = None
        self.stream_name = None
        self.in_rgb = None

    def runCamera(self):
        with dai.Device(self.pipeline) as device:
            # self.controlQueue = device.getInputQueue('control')
            #
            # self.stillQueue = device.getOutputQueue('still')

            # Output queue will be used to get the rgb frames from the output defined above
            self.q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.stream_name = "rgb-" + self.mxid + "-OAK-1"

            while doWhileLoop:
                self.in_rgb = self.q_rgb.tryGet()
                if self.in_rgb is not None:
                    cv2.imshow(self.stream_name, self.in_rgb.getCvFrame())


deviceList = []

devices = dai.Device.getAllAvailableDevices()

for device in devices:
    deviceList.append(device)

cam1 = cameraOAK(deviceList[0])
cam2 = cameraOAK(deviceList[1])

c1Thread = threading.Thread(target=cam1.runCamera())
c2Thread = threading.Thread(target=cam2.runCamera())

c1Thread.start()
c2Thread.start()

c1Thread.join()
c2Thread.join()

# q_rgb_list = []
# controlQueue_list = []
#
# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
# with contextlib.ExitStack() as stack:
#     device_infos = dai.Device.getAllAvailableDevices()
#     if len(device_infos) == 0:
#         raise RuntimeError("No devices found!")
#     else:
#         print("Found", len(device_infos), "devices")
#
#     for device_info in device_infos:
#         # Note: the pipeline isn't set here, as we don't know yet what device it is.
#         # The extra arguments passed are required by the existing overload variants
#         openvino_version = dai.OpenVINO.Version.VERSION_2021_4
#         usb2_mode = False
#         device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))
#
#         # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
#         print("=== Connected to " + device_info.getMxId())
#         mxid = device.getMxId()
#         cameras = device.getConnectedCameras()
#         usb_speed = device.getUsbSpeed()
#         print("   >>> MXID:", mxid)
#         print("   >>> Cameras:", *[c.name for c in cameras])
#         print("   >>> USB speed:", usb_speed.name)
#
#         device_type = "unknown"
#         if len(cameras) == 1:
#             device_type = "OAK-1"
#         elif len(cameras) == 3:
#             device_type = "OAK-D"
#         # If USB speed is UNKNOWN, assume it's a POE device
#         if usb_speed == dai.UsbSpeed.UNKNOWN: device_type += "-POE"
#
#         # Get a customized pipeline based on identified device type
#         pipeline = getPipeline(device_type)
#         print("   >>> Loading pipeline for:", device_type)
#         device.startPipeline(pipeline)
#
#         controlQueue = device.getInputQueue('control')
#
#         stillQueue = device.getOutputQueue('still')
#
#         # Output queue will be used to get the rgb frames from the output defined above
#         q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
#         stream_name = "rgb-" + mxid + "-" + device_type
#         # If reimplementing still frame capture, add stillQueue below
#         q_rgb_list.append((controlQueue, stillQueue, q_rgb, stream_name))
#         if verbose:
#             print(q_rgb_list)
#
# doWhileLoop = True
#
# def runCam(camControlQueue, camStillQueue, camQ_rgb, camStream_name):
#     while doWhileLoop:
#         stillFrames = camStillQueue.tryGetAll()
#         for stillFrame in stillFrames:
#             # Decode JPEG
#             frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
#             # Display
#             cv2.imshow('still', frame)
#
#         in_rgb = camQ_rgb.tryGet()
#         if in_rgb is not None:
#             cv2.imshow(camStream_name, in_rgb.getCvFrame())
#
# def keyStrokes():
#     if cv2.waitKey(1) == ord('q'):
#         global doWhileLoop
#         doWhileLoop = False
#
#     elif cv2.waitKey(1) == ord('p'):
#         cv2.imwrite(os.path.join(dirPath, f'img_{imgCount + 1}.jpeg'), frame)
#         imgCount = len([
#             f for f in os.listdir(dirPath)
#             if f.endswith('.jpeg') and os.path.isfile(os.path.join(dirPath, f))
#         ])
#         if verbose:
#             print(f'Image count = {imgCount}')
#             print(f'saved image as img_{imgCount}.jpeg')
#
#     elif cv2.waitKey(1) == ord('c'):
#         ctrl = dai.CameraControl()
#         ctrl.setCaptureStill(True)
#         controlQueue.send(ctrl)
#
#     elif cv2.waitKey(1) == ord('x'):
#         cv2.destroyWindow('still')
#
#     # Camera settings preset 1:
#     elif cv2.waitKey(1) == ord('1'):
#         for controlQueue, stillQueue, q_rgb, stream_name in q_rgb_list:
#             ctrl = dai.CameraControl()
#             cv2.setTrackbarPos('Focus', 'Camera Controls', 114)
#             cv2.setTrackbarPos('Exposure', 'Camera Controls', 1499)
#             cv2.setTrackbarPos('ISO', 'Camera Controls', 1500)
#             cv2.setTrackbarPos('White-Balance', 'Camera Controls', 1500)
#             ctrl.setManualFocus(114)
#             ctrl.setManualExposure(1500, 1600)
#             ctrl.setManualWhiteBalance(2500)
#             controlQueue.send(ctrl)
