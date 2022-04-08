from flask import Flask, render_template, Response
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Default cache expiration is 12 hours

import cv2
# camera = cv2.VideoCapture(0)
import requests

global final

from io import BytesIO
from math import log
import numpy as np
import time
import cv2
import sys

# import threading
# lock = threading.Lock()


# MAX_FILE_SIZE = 62914560 # bytes
# MAX_FILE_SIZE = int(log(MAX_FILE_SIZE, 2)+1)


# def read(buffer, number_of_bytes):
#     output = b""
#     while len(output) < number_of_bytes:
#         output += buffer.read(number_of_bytes - len(output))
#     assert len(output) == number_of_bytes, "An error occured."
#     return output

# def read_file(buffer):
#     # Read `MAX_FILE_SIZE` number of bytes and convert it to an int
#     # So that we know the size of the file comming in
#     length = int(read(buffer, MAX_FILE_SIZE))
#     # Here you can switch to a different file every time `writer.py`
#     # Sends a new file
#     data = read(buffer, length)
#     # Read a byte so that we know if it is the last file
#     file_ended = read(buffer, 1)
#     return data, (file_ended == b"1")

@app.route('/')
def main():
   return render_template('main.html')

# def gen_frames():  
#     while True:
#         # success, frame = camera.read()  # read the camera frame
#         # if not success:
#         #     break
#         frame = requests.get("http://localhost:8090/", stream=True)
#         # data, last_file = read_file(sys.stdin.buffer)
#         # frame = cv2.imdecode(np.frombuffer(BytesIO(data).read(), np.uint8), cv2.IMREAD_UNCHANGED)
#         # if frame is not None:
#         print(frame)
#         if frame.status_code == 200:
#             frame.raw.decode_content = True
#             print(frame.content)
#             ret, buffer = cv2.imencode('.jpg', frame.raw)
#             # frame = buffer.tobytes()
#             cv2.imshow("d", buffer)
#             cv2.waitKey(0)
#             yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')  # concat frame one by one and show result
#         # if last_file:
#         #     break

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
#    app.run()
   app.run(debug=True)