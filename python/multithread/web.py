from inspect import currentframe
from flask import Flask, render_template, Response

#!/usr/bin/env python3
import re
import cv2
import depthai as dai
import numpy as np
from multiprocessing.pool import ThreadPool
import threading
from queue import Queue
from collections import deque
import time

import base64
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

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

global currFrame

global final

threadn = cv2.getNumberOfCPUs()
pool = ThreadPool(processes = threadn)
pending = deque()
video = deque()

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
        # cv2.imshow(name, video.get(block=False))

def main():

    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    avgFPS = 1

    framerate = []

    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    process = deque()

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
            while len(process) > 0 and process[0].ready():
                tracked, res = process.popleft().get()
                if res is not None:
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time
                    fps = int(fps)
                    framerate.insert(0, fps)
                    avgFPS = str(int(sum(framerate)/len(framerate))+1)
                    if len(framerate) > 10:
                        del framerate[len(framerate)-1]
                    print(avgFPS)
                    # if (1/(new_frame_time-prev_frame_time) > 120):
                    #     # video.put(res)
                    #     stream = threading.Thread(target=Display.videoStream, args=("video", res))
                    #     stream.start()
                    cv2.putText(res, avgFPS, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 200, 100), 2, cv2.LINE_AA)
                    cv2.putText(tracked, avgFPS, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 200, 100), 2, cv2.LINE_AA)
                    cv2.imshow('threaded video', tracked)
                    cv2.imshow('threaded mask', res)

                    # stream = pool.apply_async(Display.videoStream, ("threaded video", res))
                    # video.append(stream)
                    # # stream = process.popleft().get()
                    currFrame = res

            if len(process) < threadn:
                inRgb = qRgb.tryGet()
                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                if frame is not None:
                    frame = frame
                    if (int(avgFPS) < 60):
                        task = pool.apply_async(Filter.procs, (frame,))
                        process.append(task)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


# @app.route('/video_feed')
# def video_feed():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')
@socketio.on("request-frame", namespace="/camera-feed")
def camera_frame_requested(message):
    frame = currFrame
    if frame is not None:
        emit("new-frame", {
            "base64": base64.b64encode(frame).decode("ascii")
        })

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, host="0.0.0.0", port=8080, threaded=True)
    # main()
    # cv2.destroyAllWindows()