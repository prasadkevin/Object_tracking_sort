# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:29:00 2020

@author: prasa
"""

"""
@author: Mahmoud I.Zidan
"""

import numpy as np
import os.path
import time
import cv2
from sort import Sort
from detector_v2 import mobile_net_ssd
import random
def createDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

total_time = 0.0
total_frames = 0

#init detector
detector = mobile_net_ssd()

#init tracker
tracker =  Sort(use_dlib= False) #create instance of the SORT tracker

net_file= '../model/people_counting_MobileNetSSD_deploy.prototxt'       
caffe_model='../model/people_counting_mobilenet_iter_216000.caffemodel'     #Mobilenet SSD caffe_model

video_name = 'TownCentreXVID.mp4'
percentage_frame_skip = 20# 20 percent of totl frame rante 

input_video = '../Data/' + video_name
cap = cv2.VideoCapture(input_video)
w,h = 720, 350

fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = int((percentage_frame_skip/100) * fps) # adjusting the skip frame using fps

output_path = 'output_videos/' + video_name[:-4] 
createDir(output_path)
output_file = output_path + '/{}%_{}_'.format(percentage_frame_skip, frame_skip) + video_name[:-4] + '.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_video_writer = cv2.VideoWriter(output_file,fourcc, 25, (w,h), True) 

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
net = cv2.dnn.readNet(caffe_model, net_file)

frame =-1
while True:

    ret, img = cap.read() 
    frame +=1
    img = cv2.resize(img, (w,h))
#    if frame < 2500:
#        continue
#    if frame == 2500:
#        break
    if frame%frame_skip ==0:
        detections = detector.detections((img, net))
#        print('detections>>>>>>>>>>>>>',list(detections))

    start_time = time.time()
    #update tracker
    trackers = tracker.update(np.array(list(detections)),img)
#    print('trackers detections>>>>>>>>>>>>>',list(trackers))
#    print('\n\n')
    cycle_time = time.time() - start_time
    total_time += cycle_time

    print('frame: %d...took: %3fs'%(frame,cycle_time))
    
    for box in trackers:
        box = box.astype(int)
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        p3 = (max(p1[0], 15), max(p1[1], 15))
        random.seed(box[4])
        colors = (random.randint(0,255),random.randint(0,200),random.randint(0,255)) 
        cv2.rectangle(img, p1, p2, colors, 2)
        img_title = "ID no. " + str(box[4])

        p3 = (p3[0], p3[1]-5)
        
        cv2.putText(img, img_title, p3, cv2.FONT_ITALIC,0.3, colors, 1,cv2.LINE_AA)
    cv2.putText(img, str(len(trackers)), (w-100, 50), cv2.FONT_ITALIC, 1, (0, 255, 255), 3,cv2.LINE_AA)
    out_video_writer.write(img)
    cv2.imshow('output', img)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
out_video_writer.release()    
             

