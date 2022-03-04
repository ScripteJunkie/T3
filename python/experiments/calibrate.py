#!/usr/bin/env python3

from typing import final
import cv2
import depthai as dai
import contextlib
import time
import numpy as np

captured, calibrated = False, False
cal_img = []

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
            # cv2.imshow("frame1", cal_img[0])
            # cv2.imshow("frame2", cal_img[1])
            img1 = cal_img[0]
            img2 = cal_img[1]
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
            # We select only inlier points
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]
            def drawlines(img1,img2,lines,pts1,pts2):
            # img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines
                r,c,d = img1.shape
                # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
                # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
                for r,pt1,pt2 in zip(lines,pts1,pts2):
                    color = tuple(np.random.randint(0,255,3).tolist())
                    x0,y0 = map(int, [0, -r[2]/r[1] ])
                    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
                    img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
                    img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
                return img1,img2
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
            lines1 = lines1.reshape(-1,3)
            img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
            lines2 = lines2.reshape(-1,3)
            img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
            # final = np.hstack((img5,img3))
            final = np.hstack((img1,img2,img3,img4,img5,img6))
            # final = np.hstack((img1,img2,img3,img4,img5))
            final = cv2.resize(final,None,fx=0.3,fy=0.3)
            cv2.imshow("Final", final)
            # cv2.imshow("frame1", img5)
            # cv2.imshow("frame2", img3)
            calibrated = True

        for q_rgb, stream_name in q_rgb_list:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                if captured == False:
                    time.sleep(3)
                    cal_img.append(in_rgb.getCvFrame().copy())
                # cv2.imshow(stream_name, in_rgb.getCvFrame())
        captured = True
        if cv2.waitKey(1) == ord('q'):
            break