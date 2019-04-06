import numpy as np
import cv2
import os


indicator_cascade = cv2.CascadeClassifier('indicator.xml')
#indicator_cascade = cv2.CascadeClassifier('myfacedetector.xml')
#frame = cv2.imread(os.path.join("indication/1322.png"))
frame = cv2.imread(os.path.join("detection_train/indication/1322.png"))

#frame = cv2.imread(os.path.join("a1.bmp"))
#this is the cascade we just made. Call what you want
# add this
# image, reject levels
# level weights.

print(indicator_cascade)


indicator = indicator_cascade.detectMultiScale(frame)
print(indicator)

# add this
for (x,y,w,h) in indicator:


    center = (x + w//2, y + h//2)
    frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 2)


cv2.imshow('img',frame)


k = cv2.waitKey(0)


cv2.destroyAllWindows()

