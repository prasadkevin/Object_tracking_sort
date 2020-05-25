# -*- coding: utf-8 -*-

"""
Created on Thu Jul  4 13:59:13 2019

--dir_name=validation/images/ --test_csv=validation/csv/all.csv --caffe_model=model/mobilenet_iter_10470.caffemodel --proto_file=model/MobileNetSSD_deploy.prototxt
"""

import numpy as np  
import cv2, glob, ntpath, os, argparse, csv
from datetime import datetime


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def createDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
def run_mobilenet_ssd(img, net):  

   h = img.shape[0]
   w = img.shape[1]

   img = img - 127.5
   img = img * 0.007843
   img = img.astype(np.float32)
   
   blob = cv2.dnn.blobFromImage(img, 1.0, (512, 512), swapRB=False)

   net.setInput(blob)
   objectsModelPreds = net.forward()
   
   clsn = objectsModelPreds[0,0,:,1]
   conf = objectsModelPreds[0,0,:,2]
   box = objectsModelPreds[0,0,:,3:7]* np.array([w, h, w, h])
   
   box, conf, cls_name = (box.astype(np.int32), conf, clsn)
   
   return box, conf, cls_name

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def create_label_bbox(out_bbox, out_conf, out_label):
    
    
    label_bbox = {}
    label_conf = {}
    
    for i in range(len(out_bbox)):
        
        label_bbox.setdefault(out_label[i], [])
        label_bbox[out_label[i]].append(out_bbox[i])
        
        label_conf.setdefault(out_label[i], [])
        label_conf[out_label[i]].append(out_conf[i])
   
    return label_bbox, label_conf 



    
   # file_names = glob.glob('personal_pics/*.JPG')           #input test images with class label as file name
#net_file= 'model/people_counting_MobileNetSSD_deploy.prototxt'       
#caffe_model='model/people_counting_atm_mobilenet_iter_90000.caffemodel'     #Mobilenet SSD caffe_model
#
#if not os.path.exists(caffe_model):
#    print(caffe_model + " does not exist")
#    exit()
#if not os.path.exists(net_file):
#    print(net_file + " does not exist")
#    exit()
#
#net = cv2.dnn.readNet(caffe_model, net_file)

class mobile_net_ssd:
    
    def detections(x_ , df):
        full_img , net = df
        
#        full_img = 'image from the filter'
        
        CLASSES = ('background', 'Person')
        threshold_dict = { 'Person': .3 , 'background' : .5}
        
        box, conf, cls_name = run_mobilenet_ssd(full_img.copy(), net)
        actual_bbox = []
        for index in range(len(box)):             
            
            label_name = CLASSES[int(cls_name[index])]  
            if conf[index] >= threshold_dict[label_name]:
                print(box[index], conf[index])
                actual_bbox.append(box[index])
        return actual_bbox
         
