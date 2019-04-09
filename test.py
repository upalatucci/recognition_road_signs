import os
import sys
import numpy as np
import cv2 as cv
from rect_intersect import Rectangle

TEST_DIR = "test_set"
TESTS_NAME = ["day", "fog", "night"]
RESULTS_DIRECTORY = "results"
SIGNS_TYPE_NAME = ["indication", "warnings", "prohibitory"]

'''
    argomenti : xml classifier path, scale Factor, minNeighbors

    True Positive : Corretto. Segnale trovato
    True Negative: Corretto. Nessun Segnale
    False Positive: Sbagliato. Segnale non trovato
    False Negative: Sbagliato. Segnale trovato dove non c'è. 
'''

def detect_road_signs(classifier, image_path, image_name, test_type):
    frame=cv.imread(image_path)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    result = classifier.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5, minSize=(20, 20), maxSize=(300, 300))

    if(len(result)>=1):
        for i in result:
            x, y, w, h = i
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),127)
    cv.imwrite("./results/"+ test_type + "/" + image_name,frame)
    return result

def test():
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    cascade = cv.CascadeClassifier(sys.argv[1])
    signs_type = SIGNS_TYPE_NAME[int(sys.argv[2])]

    total_score = [0, 0, 0, 0] # true positive, true negative , false positive, false negative
    for test_type in TESTS_NAME:
        results_test_dir = os.path.join(RESULTS_DIRECTORY, test_type)
        if not os.path.exists(results_test_dir):
            os.mkdir(results_test_dir)
        test_score = [0, 0, 0, 0] # true positive, true negative, false positive, false negative per il test
        directory_test = os.path.join(TEST_DIR, test_type)
        annotation_file = os.path.join(directory_test, "annotations.txt")
        
        ground_truth = {}
        with open(annotation_file) as annotations:
            lines = annotations.readlines()
            for line in lines:
                info = line.split(" ")
                
                if info[5].rstrip() == signs_type:
                    if info[0] not in ground_truth:
                        ground_truth[info[0]] = []
                        ground_truth[info[0]].append([info[1], info[2], info[3], info[4]])
                    else:
                        ground_truth[info[0]].append([info[1], info[2], info[3], info[4]])
            
        images = [f for f in os.listdir(directory_test) if f.split(".")[-1] in ["png", "jpg", "jpeg"]]
        

        for image in images:
            result = detect_road_signs(cascade, os.path.join(directory_test,image), image, test_type)
            if image in ground_truth:
                for bounding_box in result:
                    x, y, w, h = bounding_box
                    result_bb = Rectangle(int(x), int(y), int(w)+int(x), int(h)+int(y))

                    found_intersection = False
                    print(result_bb)
                    for bb_gt in ground_truth[image]:
                        x, y, x1, y1 = bb_gt
                        gt_rect = Rectangle(int(x), int(y), +int(x1), int(y1))
                        intersection = result_bb.intersection(gt_rect)

                        if intersection is not None and intersection.area() / result_bb.area_of_union(gt_rect) > 0.5:
                            ground_truth[image].remove(bb_gt)
                            found_intersection = True
                            break
                    
                    if found_intersection == True:
                        total_score[0] += 1
                        test_score[0] += 1
                    else:
                        total_score[2] += 1
                        test_score[2] += 1

                total_score[2] += len(ground_truth[image])
                test_score[2] += len(ground_truth[image])
            else:
                if len(result) == 0:
                    test_score[1] += 1
                    total_score[1] += 1
                test_score[3] += len(result)
                total_score[3] += len(result)
            
        print(test_score)
        test_precision = test_score[0] / (test_score[0] + test_score[2])
        test_recall = test_score[0] / (test_score[0] + test_score[3])

        test_f1score = 2 * test_precision * test_recall / (test_precision + test_recall)
    
        print("Score for " + test_type)
        print("Precision : " + str(test_precision))
        print("Recall : " + str(test_recall))
        print("F1 score : " + str(test_f1score))

    
    precision = total_score[0] / (total_score[0] + total_score[2])
    recall = total_score[0] / (total_score[0] + total_score[3])

    f1score = 2 * precision * recall / (precision + recall)

    print("Total Score")
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1score))
    print(total_score)

if __name__ == '__main__':
    test()
        
    