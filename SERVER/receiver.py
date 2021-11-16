import sys
sys.path.append('/radu')
import os
from bot import Bot
from rusb import USB
from _thread import start_new_thread
from time import sleep_ms
radu = Bot('Radu MK1')
usb = USB()
input_msg = None
bufferSTDINthread = start_new_thread(usb.bufferSTDIN, ())
while True:
  input_msg = usb.getLineBuffer()
  if input_msg and 'ros_msg' in input_msg:
    obj = eval(input_msg)
    radu.notify(obj)
  sleep_ms(10)