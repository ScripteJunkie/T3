from threading import Thread
import cv2
import depthai as dai

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

class VideoGet:
    # """
    # Class that continuously gets frames from a VideoCapture object
    # with a dedicated thread.
    # """

    def __init__(self, src=0):
        frame = None
        with dai.Device(pipeline) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:
            inRgb = qRgb.tryGet()
            if inRgb is not None:
                # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                self.frame = inRgb.getCvFrame()
            if frame is not None:
                # self.stream = cv2.VideoCapture(src)
                (self.grabbed, self.frame) = self.frame, True#stream.read()
                self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.frame, True

    def stop(self):
        self.stopped = True