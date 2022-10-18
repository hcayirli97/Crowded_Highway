import re
import cv2 ,torch,sys
import numpy as np
sys.path.append("libs/")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import letterbox

def detect(im ,model, device,imgsz):
    global names
    Frame = im.copy()

    im = letterbox(im,imgsz,stride=64,auto = False)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)

    im = im.float()
    im /= 255
    im = torch.unsqueeze(im, 0)
    pred = model(im, augment=False, visualize=False)

    pred = non_max_suppression(pred, 0.1, 0.1, None, False, max_det=50)


    Output = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], Frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c == 7 or c == 2:
                    x1 , y1 , x2 , y2 = xyxy
                    Output.append([int(x1.item()),int(y1.item()),int(x2.item()),int(y2.item()),names[c]])
    return Output

def draw(output,img):
    for out in output:
        x1,y1,x2,y2,cls = out
        cv2.circle(img, ((x1+x2)//2,(y1+y2)//2), 6, (0,255,0), -1)
        # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    car_count = len(output)
    h , w , _ = img.shape
    if car_count > 0 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.90) , int(w*0.92) , int(h*0.86)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)
    if car_count > 5 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.85) , int(w*0.92) , int(h*0.81)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)
    if car_count > 10 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.80) , int(w*0.92) , int(h*0.76)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)
    if car_count > 15 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.75) , int(w*0.92) , int(h*0.71)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)
    if car_count > 20 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.70) , int(w*0.92) , int(h*0.66)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)
    if car_count > 25 :
        b2x,b2y,b1x,b1y = int(w*0.98) , int(h*0.65) , int(w*0.92) , int(h*0.61)
        cv2.rectangle(img, (b1x,b1y) , (b2x,b2y) , (0,0,255) ,-1)
        cv2.rectangle(img, (b1x+5,b1y+5) , (b2x-5,b2y-5) , (0,0,225) , -1)

    cv2.putText(img, "Car Count: {}".format(car_count),(int(w*0.8),int(w*0.05) ),cv2.FONT_HERSHEY_DUPLEX,1,(0, 0, 0), 6)
    cv2.putText(img, "Car Count: {}".format(car_count),(int(w*0.8),int(w*0.05) ),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255), 2)
    return img      
