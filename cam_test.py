# import the necessary packages

import logging
import signal
import time
import uuid
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

FORMAT = "[%(levelname)s] %(asctime)s : %(message)s"
logging.basicConfig(filename='./logs', level=logging.DEBUG, format=FORMAT)
log = logging.getLogger(__name__)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


class SignalHandler:
    stop = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.safe_stop)
        signal.signal(signal.SIGTERM, self.safe_stop)

    def safe_stop(self, signum, frame):
        log.debug("Safe stop process %s", signum)
        self.stop = True


signal_handler = SignalHandler()

# initialize the camera and grab a reference to the raw camera capture
width = 320
height = 240
camera = PiCamera()
camera.resolution = (width, height)
camera.framerate = 30
#camera.rotation = 180
camera.color_effects = (128, 128)
rawCapture = PiRGBArray(camera, size=(width, height))

# allow the camera to warmup
time.sleep(0.1)

# Inicia DNN
prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#Substractor de fondo
#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#Elementos estructurantes para filtros morfologicos
#kernelOp = np.ones((3,3),np.uint8)
#kernelOp2 = np.ones((5,5),np.uint8)
#kernelCl = np.ones((11,11),np.uint8)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        image = frame.array

        #log.debug("Aplica substraccion de fondo")
        #fgmask = fgbg.apply(image)

        blob = cv2.dnn.blobFromImage(image, 0.007843, (width, height), 127.5)
        net.setInput(blob)
        detections = net.forward()

        log.debug("Positive detections %s", detections.shape[2])

#        # loop over the detections
#        for i in np.arange(0, detections.shape[2]):
#            # extract the confidence (i.e., probability) associated
#            # with the prediction
#            confidence = detections[0, 0, i, 2]
#
#            if confidence >= 0.65:
#                idx = int(detections[0, 0, i, 1])
#
#                # for the object
#                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
#                (startX, startY, endX, endY) = box.astype("int")
#                # Log detection
#                log.debug("{} detected at {},{}".format(CLASSES[idx], startX, startY))
#
#                image = cv2.rectangle(image,(startX, startY),(endX, endY),(0,255,0),2)
#                cv2.imwrite("./objects/{}-x{}-y{}.jpg".format(CLASSES[idx], startX, startY), image)


#        #log.debug("Binarizacion para eliminar sombras (color gris)")
#        try:
#            mask = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)[1]
#            #Opening (erode->dilate) para quitar ruido.
#            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOp)
#            #Closing (dilate -> erode) para juntar regiones blancas.
#            mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
#        except:
#            break
#
#        #log.debug("Busca contornos")
#        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#        for cnt in contours:
#            area = cv2.contourArea(cnt)
#            if area > (width*height/60) and area < (width*height/20):
#                M = cv2.moments(cnt)
#                cx = int(M['m10']/M['m00'])
#                cy = int(M['m01']/M['m00'])
#                x, y, w, h = cv2.boundingRect(cnt)
#                
#                # cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
#                image = cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)            
#                # image = cv2.drawContours(image, cnt, -1, (0,255,0), 3)
#
#                # cv2.imwrite("./objects/{}-{}-{}.jpg".format(x, y, area), image)
#
#                log.debug("Object detected at {},{} with: {}".format(x, y, area))


        #log.debug("Show frame")
        # show the frame
        #cv2.imshow("Preview", image)


        #log.debug("Clear camera buffer")

        # clear the stream in preparation for the next frame
        rawCapture.seek(0)
        rawCapture.truncate()

        # if the `q` key was pressed, break from the loop
        if signal_handler.stop or cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
