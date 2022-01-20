#!/usr/bin/env python3
import re
import cv2
import depthai as dai
import numpy as np
from multiprocessing.pool import ThreadPool
# from multiprocessing 
from collections import deque
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(1920, 1080)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

global first_iter
global points

threadn = cv2.getNumberOfCPUs()
pool = ThreadPool(processes = threadn)
pending = deque()

class Filter():
    def procs(frame):

        hsv = pool.apply_async(Filter.hsv, (frame,))
        pending.append(hsv)
        hsv = pending.popleft().get()

        # Threshold the HSV image to get only white colors
        mask = pool.apply_async(Filter.mask, (hsv,))
        pending.append(mask)
        mask = pending.popleft().get()
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        contours = pool.apply_async(Filter.contours, (mask,))
        pending.append(contours)
        contours = pending.popleft().get()

        #sorting the contour based of area
        tracked = frame.copy()
        return tracked, res

    def hsv(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv
    
    def mask(hsv):
        lower_white = np.array([2, 48, 125], dtype=np.uint8)
        upper_white = np.array([81, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask

    def contours(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

class Display():
    def videoStream(name, frame):
        cv2.imshow(name, frame)


def main():

    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    framerate = []

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        print('Connected cameras: ', device.getConnectedCameras())
        # Print out usb speed
        print('Usb speed: ', device.getUsbSpeed().name)

        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        print(np.shape(qRgb))
        frame = None

        first_iter = True
        points = []

        while True:
            while len(pending) > 0 and pending[0].ready():
                tracked, res = pending.popleft().get()
                if res is not None:
                    frame = frame
                    try:
                        new_frame_time = time.time()
                        fps = 1/(new_frame_time-prev_frame_time)
                        prev_frame_time = new_frame_time
                        fps = int(fps)
                        framerate.insert(0, fps)
                        avgFPS = str(int(sum(framerate)/len(framerate)))
                        if len(framerate) > 10:
                            del framerate[len(framerate)-1]
                        print(avgFPS)
                    except ZeroDivisionError:
                        print(framerate)
                    # cv2.putText(res, avgFPS, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 200, 100), 2, cv2.LINE_AA)
                    # cv2.putText(tracked, avgFPS, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 200, 100), 2, cv2.LINE_AA)
                    # cv2.imshow('threaded video', tracked)
                    # cv2.imshow('threaded mask', res)
                    # stream = pool.apply_async(Display.videoStream, ("threaded video", res))
                    # pending.append(stream)
                    # stream = pending.popleft().get()


            if len(pending) < threadn:
                inRgb = qRgb.tryGet()
                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                if frame is not None:
                    frame = frame
                    task = pool.apply_async(Filter.procs, (frame,))
                    pending.append(task)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()