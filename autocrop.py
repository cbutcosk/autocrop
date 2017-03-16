import cv2
from PIL import Image
import numpy as np
import imutils
import argparse

# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--template", required = True, help = "Template image" )
parser.add_argument("-i","--image", required = True, help = "Image for matching")
parser.add_argument("-o","--output", required = True, help = "Output file")
args = vars(parser.parse_args())

# Load big and small images into cv2
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50,200)
(tH, tW) = template.shape[:2]

target = cv2.imread(args["image"]) 
gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
found = None

for scale in np.linspace(0.2, 1.0, 20)[::-1]:
  resized = imutils.resize(gray_target, width = int(gray_target.shape[1] * scale))
  r = gray_target.shape[1] / float(resized.shape[1])

  if resized.shape[0] < tH or resized.shape[1] < tW:
    break
  
  edged = cv2.Canny(resized, 50,200)
  result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
  (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

  if found is None or maxVal > found[0]:
    found = (maxVal, maxLoc, r)

(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# Now take the box and crop
cropped = target[startY:endY, startX:endX]
cv2.imwrite(args["output"],cropped)
