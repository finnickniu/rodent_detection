""
import os
import gc
import cv2
import json
import torch
import sys
import numpy as np
from collections import Counter
from PIL import Image , ImageDraw
import torch.multiprocessing as mp
from itertools import combinations
from datetime import datetime
from time import gmtime, strftime
from datetime import datetime,time,timedelta
from numba import jit
from mmdet.apis import inference_detector, init_detector
import re
import time
import base64
import argparse


"""----------constant--------"""



class Server(object):
    def __init__(self,args):
        self.headers={'Content-type':'application/json', 'Accept':'application/json'}
        self.model_path="http://119.81.130.181/model.pth"
        self.label_path = "meta.json"
        self.config = "configs/cascade_rcnn/cascade_rat.py"
        self.video_dir = args.video_path
        self.sv_dir = args.ann_dir
    def box2yolo(self,box, image_width, image_height):
        object_class = 0 
        x1,y1,x2,y2 = box
        x = (x1+x2)/2/image_width
        y = (y1+y2)/2/image_height
        w = abs(x1-x2)/image_width
        h = abs(y1-y2)/image_height

        return x,y,w,h



    def model_rcnn(self,device,model_path):
        device = "cuda:0"
        model = init_detector(self.config , model_path, device=device)
        return model

    def read_label(self):
        class_file = json.load(open(self.label_path, 'r'))
        class_map = [clss["title"] for clss in class_file["classes"]]
        return class_map

    @torch.no_grad()
    def detection_inference(self,model,image):
        result = inference_detector(model, image)
        bbox_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        return bboxes,labels

    def write2txt(self,coor,filename,time_period):
        x,y,w,h = coor
        file_name = self.video_dir[:-4] + f"_{time_period}.txt"
        pattern = re.compile(r'@.+')  
        file_name = pattern.findall(file_name)[0]
        file = open(self.sv_dir+file_name, "a") 
        file.write(f"{0} {x} {y} {w} {h}"+"\n") 
        print(file_name)
        return

        


    def run(self):
       
        device = torch.device('cuda:0')
    
        model_detection = self.model_rcnn(device,self.model_path)        
        suspecious_list=[]
        frame =0 
        coor_id={}
        repeat_id=[]
        cap = cv2.VideoCapture(self.video_dir)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cnt = 1 
        while True:
            dets =[]
            time_period = cnt/fps*1000

            success,image = cap.read()
            image_height,image_width = image.shape[:2]
            # if not success: 
            #     continue
            image_copy = image.copy()
            labels = self.read_label()
            output,label = self.detection_inference(model_detection,image)
            for i in range(len(output)):
                label_name = labels[label[i]]
                result =output[i]
                score = result[-1]
                
                x1, y1, x2, y2 = int(result[0]),int(result[1]),int(result[2]),int(result[3])
                if score > 0.95:
                    
                    cv2.putText(image,f'{label_name}|{str(score)[:4]}',(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
                    x,y,w,h = self.box2yolo([x1, y1, x2, y2], image_width, image_height)
                    self.write2txt([x,y,w,h],self.video_dir,int(time_period))
                   
                    # print(file_name)

            cnt+=1

        
                # else:
                #     sys.exit()
                #     break



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AppTech Rodent Detection')
    parser.add_argument(
        '--video_path', type=str, default="@20000102013000_20000102015959_9359.mp4", help='video dir')
    parser.add_argument(
        '--ann_dir', type=str, default="ann/", help='bbox score threshold')
    args = parser.parse_args()

    mp.set_start_method('forkserver', force=True)

    #time = (start_time, end_time,email_sent_time)
    sever = Server(args)

    sever.run()
    #sever.main(cam_dir="rtsp:root:pass@192.168.1.90/axis-media/media.amp?",top=50,show=True)







    
    
