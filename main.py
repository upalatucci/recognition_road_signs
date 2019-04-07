import cv2
import os

def create_txt(dir,file_name):
    path= os.listdir("./detection_train/"+dir)
    with open(file_name, mode='w') as txtfile:
        for i in path:
            img=cv2.imread("./detection_train/"+dir+"/"+i)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            dim=img.shape
            result="rawdata"+"/"+i+" 1 0 0 "+str(dim[0])+" "+str(dim[1])+"\n"
            #equalized=0
            #equalized = cv2.equalizeHist(grey,equalized)
            #cv2.imshow("",equalized)

            txtfile.write(result)


if __name__=='__main__':

    create_txt("indication", "info.txt")
