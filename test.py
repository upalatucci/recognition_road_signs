import os
import sys
import numpy as np
import cv2 as cv

TEST_DIR = "test_set"
TESTS_NAME = ["day", "fog", "night"]
RESULTS_DIRECTORY = "results"
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
    result = classifier.detectMultiScale(gray, scaleFactor=float(sys.argv[2]), minNeighbors=int(sys.argv[3]), minSize=(20, 20), maxSize=(300, 300))

    if(len(result)>=1):
        for i in result:
            x, y, w, h = i
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),127)
    cv.imwrite("./results/"+ test_type + "/" + image_name,frame)
    return result

def test(classifier_path):
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    indicator_cascade = cv.CascadeClassifier(classifier_path)

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
                if info[0] not in ground_truth:
                    ground_truth[info[0]] = 1
                else:
                    ground_truth[info[0]] += 1
            
        images = [f for f in os.listdir(directory_test) if f.split(".")[-1] in ["png", "jpg", "jpeg"]]


        for image in images:
            result = detect_road_signs(indicator_cascade, os.path.join(directory_test,image), image, test_type)
            if image in ground_truth:
                delta = ground_truth[image] - len(result)

                if delta == 0:
                    test_score[0] += len(result)
                    total_score [0]+= len(result)
                elif delta > 0:
                    test_score[0] += len(result)
                    total_score [0]+= len(result)
                    test_score[2] += delta
                    total_score[2] += delta
                elif delta < 0:
                    test_score[0] += ground_truth[image]
                    total_score[0] += ground_truth[image]
                    test_score[3] -= delta
                    total_score[3] -= delta
            else:
                if len(result) == 0:
                    test_score[1] += 1
                    total_score[1] += 1
                test_score[3] += len(result)
                total_score[3] += len(result)
            
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
    test(sys.argv[1])
        
    