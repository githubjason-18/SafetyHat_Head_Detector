 
 
import cv2
import numpy as np 
import argparse
import time
from imutils.video import FPS
from imutils.video import VideoStream

fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('new.avi', fourcc, 30,   (800, 600), True)
def load_yolo():
    net = cv2.dnn.readNet("yolov4-train_final.weights", "yolov4-train.cfg")
    classes = []
    with open("classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes,output_layers
 


    
def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                
                
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids
   
def get_box_color(imageFrame):
    
    
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
  
    # Set range for red color and 
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
  
    # Set range for green color and 
    # define mask
    green_lower = np.array([40, 40, 40], np.uint8)
    green_upper = np.array([86, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    yellow_lower = np.array([25, 146, 190], np.uint8)
    yellow_upper = np.array([62, 174, 250], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    """
    white_lower = np.array([20, 100, 100], np.uint8)
    white_upper = np.array([30, 255, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
     """
    orange_lower = np.array([30, 50, 50], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)
    
    white_lower = np.array([0, 0, 0], np.uint8)
    white_upper = np.array([0,0,255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
      
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = red_mask)
      
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)
      
    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = blue_mask)
     
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = yellow_mask)
     
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = orange_mask)
    white_mask = cv2.dilate(white_mask, kernal)
    res_white= cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = white_mask)
    c=(255,255,255) 
    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((0, 0, 255))
           
    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((0,255,0))
    
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((255,0,0))
    
    contours, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((0,165,255))
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((51,255,255)) 
    contours, hierarchy = cv2.findContours(white_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            return((255,255,255))
    # Creating contour to track blue color
    
    

    # Program Termination
    
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    return(c)


def draw_labels(boxes, confs, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    for i in range(len(boxes)):
        if i in indexes:
            print(i)
            x, y, w, h = boxes[i]
            if(x<0): x=0
            if(y<0): y=0
            
            color=(0,0,0)
            nimg=img[y:y+h,x:x+w,:]
            
                
            
            
            
            color=get_box_color(nimg)
            
            label = classes[class_ids[i]]
            text = "{}: {:.4f}%".format(label, (confs[i]*100))
            
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, text, (x, y - 5), font, 1, color, 1)
    writer.write(cv2.resize(img,(800, 600)))
    fps.update()
    
 

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels





def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def start_video(video_path):
    model, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        #colors=np.array(colors)
        draw_labels(boxes, confs, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
start_video('video.mp4')








