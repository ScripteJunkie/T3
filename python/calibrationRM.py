import numpy as np
import cv2
import depthai as dai
import contextlib
# first do: pip install future
# tkinter should work after this
import tkinter as tk

def nothing(x):
    pass


cv2.namedWindow('Camera Controls')

cv2.createTrackbar('focus', 'Camera Controls', 0, 255, nothing)

# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # For the demo, just set a larger RGB preview size for OAK-D
    if device_type.startswith("OAK-D"):
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setPreviewSize(1920, 1080)
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
        if len(cameras) == 1: device_type = "OAK-1"
        elif len(cameras) == 3: device_type = "OAK-D"
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
        q_rgb_list.append((controlQueue, stillQueue, q_rgb, stream_name))

    while True:
        exposure = cv2.getTrackbarPos("Exposure", "Trackbars")
        iso = cv2.getTrackbarPos("ISO", "Trackbars")
        focus = cv2.getTrackbarPos("focus", "Camera Controls")

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
                ctrl = dai.CameraControl()
                ctrl.setManualFocus(focus)
                ctrl.setManualExposure(exposure, iso)
                controlQueue.send(ctrl)

        if cv2.waitKey(1) == ord('q'):
            break

        elif cv2.waitKey(1) == ord('p'):
            cv2.imwrite('test.jpeg', frame)

        elif cv2.waitKey(1) == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)