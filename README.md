# SafetyHat_Head_Detector
The project implements YOLO v4 to detect head and safety hats in real time. It also predicts the color of the Safety Hat and draw the bounding box of the corresponding color. 
YOLO v4 is implemented on Google Colab and yolov4_final_weights file is generated. This weight file was used to compute the outputs and draw labels and bounding box on the input given. The input can be an image, webcam feed or a video. YOLO v4 is ultra fast and works on COCO dataset with real time speed of 65 FPS.
The project detects two classes
0 Safety Hat
1 Head

The dataset used was downloaded from https://wobotintelligence-my.sharepoint.com/personal/animikh_wobot_ai/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fanimikh%5Fwobot%5Fai%2FDocuments%2FHackathon%2FDataset&originalPath=aHR0cHM6Ly93b2JvdGludGVsbGlnZW5jZS1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9hbmltaWtoX3dvYm90X2FpL0Vnc3hvcTV6YW1wUGpmbm9KbFB6X3owQkFyODdLWGttSXZZNkRLclRDTnltYWc%5FcnRpbWU9Zi1IWURCWkYyVWc 

Since, the annotations are in .xml formal, it can be converted into .txt format using the file convertxmltotxt.py I have provided.
The Google Colab has been provided to train your own Custom Dataset with YOLO v4.

OpenCV was used to predict the color of the Safety Hat. 
The function get_bounding_color() was created to get the color of Safety Hat.
Each image/frame was converted into HSV format.
The upper and lower HSV values of all conventional colours of Safety hats were used to create corresponding masks.
The contours were drawn and if a color with area>300 was present, the BGR value of the color was returned.

Then bounding boxes, labels and corresponding confidence values are drawn on the output image/frame.
We get a demo output as this one-
![Demo](https://user-images.githubusercontent.com/47152563/125267382-5dfb2500-e324-11eb-94c0-0c1ffed7f457.mp4)
