from ast import Break
from email.mime import image
import cv2 ,torch,sys,glob
import numpy as np
sys.path.append("libs/")
from get_model import get_model
from detect import detect ,draw



weights = "models\yolov5x.pt"
data = "C:/Users\Huseyin\Desktop\Workspace\Poker\yolov5-master\data\coco128.yaml"
imgsz = [640,640]
device = torch.device("cuda:0")
model = get_model(weights,device,data,imgsz)


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter("output.mp4", fourcc, 10.0, (1280,720))

cap = cv2.VideoCapture('highroad_new.mp4')

counter = 0
while(True):
    ret, img = cap.read()
    if ret == True:

        output = detect(img ,model, device,imgsz)
        

        counter += 1
        out_frame = draw(output,img)

        # cv2.imshow("",out_frame)
        # cv2.waitKey(1)
        video.write(out_frame)
        print("\rCounter: {}".format(counter), end = "")
    else:
        break
video.release()


  
