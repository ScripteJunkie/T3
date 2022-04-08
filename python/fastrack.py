#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time
import imutils
from collections import deque
from envHandler import toList
import csv
import argparse
from multiprocessing import Queue
import ray
ray.init()


import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from time import sleep
from socketserver import ThreadingMixIn
import threading
from PIL import Image 

import contextlib
from math import log
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', type=int, help="Index of camera. (1 or 2)")
args = parser.parse_args()
camNum = args.camera


MAX_FILE_SIZE = 62914560 # bytes
MAX_FILE_SIZE = int(log(MAX_FILE_SIZE, 2)+1)


def write_file(buffer, data, last_file=False):
   # Tell `reader.py` that it needs to read x number of bytes.
   length = len(data)
   # We also need to tell `read.py` how many bytes it needs to read.
   # This means that we have reached the same problem as before.
   # To fix that issue we are always going to send the number of bytes but
   # We are going to pad it with `0`s at the start.
   # https://stackoverflow.com/a/339013/11106801
   length = str(length).zfill(MAX_FILE_SIZE)
   with open("output.txt", "w") as file:
      file.write(length)
   buffer.write(length.encode())

   # Write the actual data
   buffer.write(data)

   # We also need to tell `read.py` that it was the last file that we send
   # Sending `1` means that the file has ended
   buffer.write(str(int(last_file)).encode())
   buffer.flush()


HTTP_SERVER_PORT = 8090

class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        # Handle is called each time a client is connected
        # When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        # First send HTTP header
        header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        self.request.send(header.encode())
        while True:
            sleep(0.1)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


# start TCP data server
# server_TCP = socketserver.TCPServer(('localhost', (1140)), TCPServerRequest)
# th = threading.Thread(target=server_TCP.serve_forever)
# th.daemon = True
# th.start()


# start MJPEG HTTP Server
server_HTTP1 = ThreadedHTTPServer(('localhost', HTTP_SERVER_PORT+1), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP1.serve_forever)
th2.daemon = True
th2.start()

server_HTTP2 = ThreadedHTTPServer(('localhost', HTTP_SERVER_PORT+2), VideoStreamHandler)
th3 = threading.Thread(target=server_HTTP2.serve_forever)
th3.daemon = True
th3.start()

import threading
lock = threading.Lock()

frame = None
outputFrame = None

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

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

pts = deque(maxlen=64)

@ray.remote
def proc(frame, num):
    global HTTP_SERVER_PORT
    HTTP_SERVER_PORT = 8090+num
    prev_frame_time = time.time()
    # cv2.imshow("rgb", frame)
    img = imutils.resize(frame, width=1920)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, np.array(toList('BALL_HSV_MIN'), dtype=np.uint8), np.array(toList('BALL_HSV_MAX'), dtype=np.uint8))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # cv2.imshow("Mask", mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # print("x coord: ", x, "y coord: ", y)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if 200 > radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 10, (0, 0, 255), -1)
                # write the data
            with open('coord.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([x, y, time.time()])
                # server_TCP.datatosend = str([x, y, time.time()])
    # update the points queue
    pts.appendleft(center)
        # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # show the frame to our screen
    # cv2.imshow("Frame", img)
    # cv2.imshow("fr", frame)
    # with lock:
    outputFrame = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_LANCZOS4)
    # if num == 1:
    #     server_HTTP1.frametosend = outputFrame
    # else:
    #     server_HTTP2.frametosend = outputFrame
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    print(str(round(new_frame_time-prev_frame_time, 3)* 1000) + "ms " + str(round(fps, 3)) + "fps" )
    return outputFrame


with open('coord.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(["X", "Y", "TIME"])


if __name__ == '__main__':
    # Connect to device and start pipeline
    with contextlib.ExitStack() as stack:
        device_infos = dai.Device.getAllAvailableDevices()
        for device_info in device_infos:
            openvino_version = dai.OpenVINO.Version.VERSION_2021_4
            usb2_mode = False
            device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

            mxid = device.getMxId()
            cameras = device.getConnectedCameras()
            usb_speed = device.getUsbSpeed()

            pipeline = getPipeline("OAK-1")
            print("   >>> Loading pipeline for:", "OAK-1")
            device.startPipeline(pipeline)

            # Clear Queue
            device.getQueueEvents() #Might be in the wrong spot idk

            # Output queue will be used to get the rgb frames from the output defined above
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            stream_name = "rgb-" + mxid + "-" + "OAK-1"
            q_rgb_list.append((q_rgb, stream_name))

        # print('Connected cameras: ', device.getConnectedCameras())
        # Print out usb speed
        # print('Usb speed: ', device.getUsbSpeed().name)

        # Output queue will be used to get the rgb frames from the output defined above
        # qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb1 = q_rgb_list[0][0].tryGet()
            in_rgb2 = q_rgb_list[1][0].tryGet()
            # if in_rgb1 is not None:
            #     frame = in_rgb1.getCvFrame()
            #     out = proc(frame, 1)
            #     # out = Process(target=proc, args=(frame, 1)).start()
            #     server_HTTP1.frametosend = out

            # if in_rgb2 is not None:
            #     frame = in_rgb2.getCvFrame()
            #     out = proc(frame, 2)
            #     # out = Process(target=proc, args=(frame, 2)).start()
            #     server_HTTP2.frametosend = out
            if in_rgb1 is not None and in_rgb2 is not None:
                ret_id1 = proc.remote(in_rgb1.getCvFrame(), 1)
                ret_id2 = proc.remote(in_rgb2.getCvFrame(), 2)
                ret1, ret2 = ray.get([ret_id1, ret_id2])
                server_HTTP1.frametosend = ret1
                server_HTTP2.frametosend = ret2
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break