#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import time
import numpy as np
import multiprocessing

# Init 2 video streams
out1 = cv2.VideoWriter('Video1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))
out2 = cv2.VideoWriter('Video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))


# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(1920, 1080)
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
            # write frame to queue
            out1.write(frame)
            #show frame
            cv2.imshow("1", frame)
        if in_rgb2 is not None:
            frame = in_rgb2.getCvFrame()
            # write frame to queue
            out2.write(frame)
            # show frame
            cv2.imshow("2", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# write to file
out1.release()
out2.release()