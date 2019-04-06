import cv2
import os

def create_txt(dir,file_name):
    path= os.listdir("./detection_train/"+dir)
    with open(file_name, mode='w') as txtfile:
        for i in path:
            img=cv2.imread("./detection_train/"+dir+"/"+i)
            dim=img.shape
            print(dim)
            result=dir+"/"+i+" 1 0 0 "+str(dim[0])+" "+str(dim[1])+"\n"
            txtfile.write(result)


if __name__=='__main__':

    create_txt("indication", "info.txt")