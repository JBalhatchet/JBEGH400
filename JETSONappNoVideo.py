#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import itertools
from collections import deque
import time
import cv2 as cv
import mediapipe as mp

from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    
    args = get_args()
    gestures = ["Pass","Stop","Execute","One","Zero","Turn Clockwise","Turn Anticlockwise","Go Forward","Erase commands","Stop listening","Two","Three"]
    
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence


    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()


    
    history_length = 16
    point_history = deque(maxlen=history_length)

    
    
    
    commands=[]
    #buffer=0
    start=time.time()
    previousSign=''
    listening=True
    print("Initialised")
    while True:
        

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
               
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                
                if hand_sign_id == 2: 
                    point_history.append(landmark_list[8])  
                else:
                    point_history.append([0, 0])

                
                
                
                gestures = ["Pass","Stop","Execute","One","Zero","Turn Clockwise","Turn Anticlockwise","Go Forward","Erase commands","Stop listening","Two","Three"]
                
                seenSign=gestures[hand_sign_id]
                
                
                
                if seenSign != previousSign:
                    #buffer+=1
                    
                    #if buffer>20:
                    now=time.time()
                    if now-start==0.5:    
                        
                        if listening==True:
                            previousSign=seenSign
                            print(seenSign)
                            #buffer=0
                            start = time.time()
                            
                            if seenSign=='Stop listening':
                                listening=False
                            
                            else:
                                
                                commands.append(seenSign)
                                
                                if seenSign=='Erase commands':
                                    print("Erasing commands from memory")
                                    
                                    commands=[]
                                if seenSign=='Execute':
                                    print("Executing commands:")
                                    commands.remove('Execute')
                                    try:
                                        commands.remove('Pass')
                                    except:
                                        print(commands)
                                        commands=[]
                                    else:
                                        print(commands)
                                        commands=[]
                        elif listening==False and seenSign=='Pass':
                            print('Recording commands')
                            listening=True
                            #buffer=0
                            start = time.time()
                            
                            previousSign=seenSign
                         
                            
                
                

    cap.release()
    cv.destroyAllWindows()





def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list




if __name__ == '__main__':
    main()
