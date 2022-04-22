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
import ray
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from time import sleep
from socketserver import ThreadingMixIn
import threading
from PIL import Image 
import contextlib

ray.init()

# parser = argparse.ArgumentParser()
# parser.add_argument('-cam', '--camera', type=int, help="Index of camera. (1 or 2)")
# args = parser.parse_args()
# camNum = args.camera

class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        # Handle is called each time a client is connected
        # When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        # First send HTTP header
        header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        self.request.send(header.encode())
        while True:
            sleep(0.01)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.001)
            if hasattr(self.server, 'frametosend') and self.server.frametosend is not None:
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    # Pass requests to seperate thread
    pass


# start TCP data server
server_TCP1 = socketserver.TCPServer(('0.0.0.0', (1140+1)), TCPServerRequest)
th = threading.Thread(target=server_TCP1.serve_forever)
th.daemon = True
th.start()

server_TCP2 = socketserver.TCPServer(('0.0.0.0', (1140+2)), TCPServerRequest)
th2 = threading.Thread(target=server_TCP2.serve_forever)
th2.daemon = True
th2.start()

server_TCP3 = socketserver.TCPServer(('0.0.0.0', (1140)), TCPServerRequest)
th3 = threading.Thread(target=server_TCP3.serve_forever)
th3.daemon = True
th3.start()

# start MJPEG HTTP Server
server_HTTP1 = ThreadedHTTPServer(('0.0.0.0', 8090+1), VideoStreamHandler)
th4 = threading.Thread(target=server_HTTP1.serve_forever)
th4.daemon = True
th4.start()

server_HTTP2 = ThreadedHTTPServer(('0.0.0.0', 8090+2), VideoStreamHandler)
th5 = threading.Thread(target=server_HTTP2.serve_forever)
th5.daemon = True
th5.start()

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

with open('coord.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(["X", "Y", "TIME"])

q_rgb_list = []
controlQueue_list = []
prev_frame_time = 0
new_frame_time = 0
frame = None
outputFrame = None
pts1 = deque(maxlen=64)
pts2 = deque(maxlen=64)

@ray.remote
def proc(frame, num):
    # prev_frame_time = time.time()
    # cv2.imshow("rgb", frame)
    img = imutils.resize(frame, width=1920)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    if num == 2:
        pts = pts1
        mask = cv2.inRange(hsv, np.array(toList('BALL1_HSV_MIN'), dtype=np.uint8), np.array(toList('BALL1_HSV_MAX'), dtype=np.uint8))
    else:
        pts = pts2
        mask = cv2.inRange(hsv, np.array(toList('BALL2_HSV_MIN'), dtype=np.uint8), np.array(toList('BALL2_HSV_MAX'), dtype=np.uint8))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # cv2.imshow("Mask", mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    x, y = None, None
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
                # if num == 2:
                #     server_TCP2.datatosend = str([x, y, time.time()])
                # else:
                #     server_TCP1.datatosend = str([x, y, time.time()])
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
    outputFrame = img#cv2.resize(img, (1280, 720), interpolation = cv2.INTER_LANCZOS4)
    # if num == 1:
    #     server_HTTP1.frametosend = outputFrame
    # else:
    #     server_HTTP2.frametosend = outputFrame
    # new_frame_time = time.time()
    # fps = 1/(new_frame_time-prev_frame_time)
    # print(str(round(new_frame_time-prev_frame_time, 3)* 1000) + "ms " + str(round(fps, 3)) + "fps" )
    return outputFrame, [x, y, time.time()]

def to3D(side, top):
    # topview table length: 1780px, width: 971px, px to inch: 0.00525437124
    # tableClose length: 1836px, tableFar length: 1100px, closepx to inch: 
    tableClose, tableFar = 762, 674 # Y value of front and back edge of table in sideview (top:0, bottom:1080)
    tableTop, tableBottom = 54, 1024 # Y value of front and back edge of table in topview (top:0, bottom:1080)
    cntX, cntY = 982, 540 # center of table in top view
    pcentX, pcntY = 940, 508 # center of perspective in top view
    sideX, sideY = float(side[0]), float(side[1])
    topX, topY = float(top[0]), float(top[1])
    x = (top[0] - cntX) * 0.00525437124
    y = (top[1] - cntY) * 0.00525437124
    pct = (top[1]-tableTop)/(tableBottom-tableTop) # percent towards back of table
    z = ((tableClose - side[1])-(80*pct))*((pct*(0.008181818182-0.004901960784))+0.004901960784)
    return round(x, 4), round(y,4), round(z, 4)


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
            print("   >>> Loading pipeline for:", mxid)

            device.startPipeline(pipeline)
            # Clear Queue
            device.getQueueEvents() #Might be in the wrong spot idk
            # Output queue will be used to get the rgb frames from the output defined above
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            stream_name = "rgb-" + mxid + "-" + "OAK-1"
            # controlQueue = device.getInputQueue('control')
            q_rgb_list.append((q_rgb, stream_name))

        global coords
        coords = '0, 0, 0'

        while True:
            prev_frame_time = time.time()
            if q_rgb_list[0][1].tryGet() == "rgb-14442C10313247D700-OAK-1":
                in_rgb1 = q_rgb_list[0][0].tryGet()
                in_rgb2 = q_rgb_list[1][0].tryGet()
            else:
                in_rgb1 = q_rgb_list[1][0].tryGet()
                in_rgb2 = q_rgb_list[0][0].tryGet()
            if in_rgb1 is not None and in_rgb2 is not None:
                # ctrl1 = dai.CameraControl()
                # ctrl1.setManualExposure(120, 1600)
                # controlQueue.send(ctrl1)
                ret_id1 = proc.remote(cv2.rotate(in_rgb1.getCvFrame(), cv2.ROTATE_180), 1)
                # ctrl2 = dai.CameraControl()
                # ctrl2.setManualExposure(100, 1600)
                # controlQueue.send(ctrl2)
                ret_id2 = proc.remote(in_rgb2.getCvFrame(), 2)
                ret1, ret2 = ray.get([ret_id1, ret_id2])
                server_HTTP1.frametosend = ret1[0]
                server_HTTP2.frametosend = ret2[0]
                server_TCP1.datatosend = str(ret1[1])
                server_TCP2.datatosend = str(ret2[1])
                server_TCP3.datatosend = coords
                if ret1[1][1] is not None and ret2[1][1] is not None:
                    coords = str(to3D(ret1[1], ret2[1]))
                    print(to3D(ret1[1], ret2[1]))