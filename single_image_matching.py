import numpy as np
import cv2
import os



indicator_cascade = cv2.CascadeClassifier('indicator_20steps.xml')
#indicator_cascade = cv2.CascadeClassifier('myfacedetector.xml')
#frame = cv2.imread(os.path.join("detection_train/indication/92.png"))
#frame = cv2.imread(os.path.join("detection_train/prohibitory/855.png"))
#frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

#frame = cv2.imread(os.path.join("a1.bmp"))
#this is the cascade we just made. Call what you want
# add this
# image, reject levels
# level weights.

#path = os.getcwd()+"/detection_train/indication/"
#path = os.getcwd()+"/detection_train/prohibitory/"
path = os.getcwd()+"/Rectangular/"
dir= os.listdir(path)
matched = 0
unmatched = 0

for i in dir:

    frame=cv2.imread(path+i)

   # print(indicator_cascade)
    indicator = indicator_cascade.detectMultiScale(frame)
    #print(indicator)
    print(indicator,i)



    if(len(indicator)>=1):

        for i in indicator:
           # (x,y,w,h) = indicator[0]
            x=i[0]
            y=i[1]
            w=i[2]
            h=i[3]
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (0, 0, 255), 2)

            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),2)


        imS = cv2.resize(frame, (960,540))                    # Resize image (Solo per vedere meglio)
        cv2.imshow("output", imS)


        cv2.waitKey(0)
        matched = matched + 1
    else:
            unmatched = unmatched + 1
            print("Unmatched"+str(unmatched))

print("******FINALSCORE  Matched="+str(matched)+" Unmatched="+str(unmatched))
total_matching = matched/(matched+unmatched)
print("Percentuale matching= "+str(total_matching))



cv2.destroyAllWindows()

