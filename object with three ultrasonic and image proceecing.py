# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
from gtts import gTTS
import os
import RPi.GPIO as GPIO
import time
import requests 
from time import sleep
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

TRIG = 21
ECHO = 20
 
TRIG1 = 6
ECHO1 = 5

TRIG2 = 13
ECHO2 = 19
led = 22 
buzzer = 12
GPIO.setup(TRIG,GPIO.OUT)                  # initialize GPIO Pin as outputs
GPIO.setup(ECHO,GPIO.IN)                   # initialize GPIO Pin as input
 
GPIO.setup(TRIG1,GPIO.OUT)                  # initialize GPIO Pin as outputs
GPIO.setup(ECHO1,GPIO.IN)                   # initialize GPIO Pin as input

GPIO.setup(TRIG2,GPIO.OUT)                  # initialize GPIO Pin as outputs
GPIO.setup(ECHO2,GPIO.IN)                   # initialize GPIO Pin as input
GPIO.setup(led,GPIO.OUT)

GPIO.setup(buzzer,GPIO.OUT)

def ultra_sonic1():
     i=0
     avgDistance=0
     for i in range(5):
      GPIO.output(TRIG, False)                 #Set TRIG as LOW
      time.sleep(0.1)                                   #Delay

      GPIO.output(TRIG, True)                  #Set TRIG as HIGH
      time.sleep(0.00001)                           #Delay of 0.00001 seconds
      GPIO.output(TRIG, False)                 #Set TRIG as LOW
      while GPIO.input(ECHO)==0:              #Check whether the ECHO is LOW
            GPIO.output(led, False)
      pulse_start = time.time()

      while GPIO.input(ECHO)==1:              #Check whether the ECHO is HIGH
            GPIO.output(led, False)
      pulse_end = time.time()
      pulse_duration = pulse_end - pulse_start #time to get back the pulse to s$

      distance = pulse_duration * 17150        #Multiply pulse duration by 1715
      distance = round(distance,2)                 #Round to two decimal points
      avgDistance=avgDistance+distance

     avgDistance=avgDistance/5
     print(avgDistance)

     if avgDistance < 15:      #Check whether the distance is within 15 cm range
        GPIO.output(buzzer, 1)
        play_audio("front side Obstacle Detected")
     else:
        GPIO.output(buzzer, 0)     


def ultra_sonic2():
     j=0
     avgDistance1=0
     for j in range(5):
      GPIO.output(TRIG1, False)                 #Set TRIG as LOW
      time.sleep(0.1)                                   #Delay

      GPIO.output(TRIG1, True)                  #Set TRIG as HIGH
      time.sleep(0.00001)                           #Delay of 0.00001 seconds
      GPIO.output(TRIG1, False)                 #Set TRIG as LOW
      while GPIO.input(ECHO1)==0:              #Check whether the ECHO is LOW
            GPIO.output(led, False)
      pulse_start1 = time.time()

      while GPIO.input(ECHO1)==1:              #Check whether the ECHO is HIGH
            GPIO.output(led, False)
      pulse_end1 = time.time()
      pulse_duration1 = pulse_end1 - pulse_start1 #time to get back the pulse to s$

      distance1 = pulse_duration1 * 17150        #Multiply pulse duration by 1715
      distance1 = round(distance1,2)                 #Round to two decimal points
      avgDistance1=avgDistance1+distance1

     avgDistance1=avgDistance1/5
     print(avgDistance1)

     if avgDistance1 < 15:      #Check whether the distance is within 15 cm range
        GPIO.output(buzzer, 1)
        play_audio("right side Obstacle Detected")
     else:
        GPIO.output(buzzer, 0)     

def ultra_sonic3():
     k=0
     avgDistance2=0
     for k in range(5):
      GPIO.output(TRIG2, False)                 #Set TRIG as LOW
      time.sleep(0.1)                                   #Delay

      GPIO.output(TRIG2, True)                  #Set TRIG as HIGH
      time.sleep(0.00001)                           #Delay of 0.00001 seconds
      GPIO.output(TRIG2, False)                 #Set TRIG as LOW
      while GPIO.input(ECHO2)==0:              #Check whether the ECHO is LOW
            GPIO.output(led, False)
      pulse_start2 = time.time()

      while GPIO.input(ECHO2)==1:              #Check whether the ECHO is HIGH
            GPIO.output(led, False)
      pulse_end2 = time.time()
      pulse_duration2 = pulse_end2 - pulse_start2 #time to get back the pulse to s$

      distance2 = pulse_duration2 * 17150        #Multiply pulse duration by 1715
      distance2 = round(distance2,2)                 #Round to two decimal points
      avgDistance2=avgDistance2+distance2

     avgDistance2=avgDistance2/5
     print(avgDistance2)

     if avgDistance2 < 15:      #Check whether the distance is within 15 cm range
        GPIO.output(buzzer, 1)
        play_audio("left side Obstacle Detected")
     else:
        GPIO.output(buzzer, 0)


def play_audio(msg):
        mytext = msg
          
        # Language in which you want to convert
        language = 'en'
          
        # Passing the text and language to the engine, 
        # here we have marked slow=False. Which tells 
        # the module that the converted audio should 
        # have a high speed
        myobj = gTTS(text=mytext, lang=language, slow=False)
          
        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save("file.mp3")
          
        # Playing the converted file
        os.system("omxplayer file.mp3")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        ultra_sonic1()
        ultra_sonic2()
        ultra_sonic3()
        
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (224, 224), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        boxes = []
        centers = []
        confidences = []
        texts = []

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.3:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        x = int(startX - (endX / 2))
                        y = int(startY - (endY / 2))
                        confidences.append(float(confidence))
                        boxes.append([x, y, int(endX), int(endY)])
                        centers.append((startX, startY))
                        # extract the bounding box coordinates
                        try:
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (W, H) = (boxes[i][2], boxes[i][3])
                        except:
                                continue
                        label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        startX, startY = centers[i][0], centers[i][1]
                        if startX <= W/3:
                                W_pos = "left "
                        elif startX <= (W/3 * 2):
                                W_pos = "center "
                        else:
                                W_pos = "right "
                                    
                        if startY <= H/3:
                            H_pos = "top "
                        elif startY <= (H/3 * 2):
                            H_pos = "mid "
                        else:
                            H_pos = "bottom "
                        texts.append(H_pos + W_pos + CLASSES[idx])
                        
        print(texts)
        # show the output frame
        cv2.imshow("Frame", frame)
        if texts:
            finaltext = ', '.join(texts)
            #disable audio for faster speed
            play_audio(finaltext)
            
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
