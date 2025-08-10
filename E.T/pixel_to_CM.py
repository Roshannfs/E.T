import cv2
import numpy as np
import math

ratio = 10 / 137 #cm / pixels
pixel_value = int(input("Enter pixel value: "));
distance_cm = pixel_value * ratio
print(distance_cm, "cm")