import bitstring
import cv2
import flask
import numpy


print("bitstring:", bitstring.__version__)
print("cv2:", cv2.__version__)
print("flask:", getattr(flask, "__version__", "ok"))
print("numpy:", numpy.__version__)
