import cv2
import os
import sys

def create_txt():
    path= os.listdir("./detection_train/"+ sys.argv[1])
    with open(sys.argv[2], mode='w') as txtfile:
        for i in path:
            img=cv2.imread("./detection_train/"+sys.argv[1]+"/"+i)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            dim=img.shape
            result="./detection_train/"+ sys.argv[1] +"/"+i+" 1 0 0 "+str(dim[0])+" "+str(dim[1])+"\n"
            #equalized=0
            #equalized = cv2.equalizeHist(grey,equalized)
            #cv2.imshow("",equalized)

            txtfile.write(result)

    with open(sys.argv[4], mode="w") as bgfile:
        for f in os.listdir(sys.argv[3]):
            bgfile.write(sys.argv[3] + f + "\n")


if __name__=='__main__':
    create_txt()
