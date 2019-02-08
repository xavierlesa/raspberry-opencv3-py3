#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import cv2
import datetime
import logging
import numpy as np
import os
import resource
import signal
import sys
import time

from threading import Thread
try:
    from queue import Queue
except:
    from Queue import Queue

from picamera import PiCamera
from picamera.array import PiRGBArray

FORMAT = "[%(levelname)s] %(asctime)s : %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
log = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class ThreadStreamMixing:
    stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class CamVideoStream(ThreadStreamMixing):
    def __init__(self, resolution=(320, 240), framerate=32, rotation=0, grayscale=True):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.rotation = rotation

        if grayscale:
            self.camera.color_effects = (128, 128)

        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr",
                use_video_port=True)

        self.frame = None

    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)

            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return


class FileVideoStream(ThreadStreamMixing):
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()

    def update(self):
        while True:
            if self.stopped:
                return
            grabbed, frame = self.stream.read()


class VideoStream:
    def __init__(self, src=0, pi_cam=False, resolution=(320,240), framerate=32, rotation=0):

        if pi_cam:
            self.stream = CamVideoStream(resolution=resolution, framerate=framerate, rotation=rotation)
        else:
            self.stream = FileVideoStream(src)

    def start(self):
        return self.stream.start()

    def update(self):
        self.stream.update()

    def read(self):
        return self.stream.read()

    def stop(self):
        self.stream.stop()

 
class ImageShow(ThreadStreamMixing):
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def set_frame(self, frame):
        self.frame = frame

    def update(self):
        while not self.stopped:
            if self.frame is not None:
                # show the frame
                cv2.imshow("Frame", self.frame)


class ImageProcessor:

    PROCESS_HOG = 'hog'
    PROCESS_MOG = 'mog'
    PROCESS_DNN = 'dnn'


    DNN_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]


    def __init__(self, resolution, process_type='hog', **kwargs):

        self.process_type = process_type

        self.min_area = kwargs.get('min_area', 5)
        self.max_area = kwargs.get('max_area', 40)

        print("process_type %s, areas %s, %s" % (self.process_type, self.min_area, self.max_area))

        # initialize the HOG descriptor/person detector
        if self.process_type == self.PROCESS_HOG:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        elif self.process_type == self.PROCESS_MOG:
            #Substractor de fondo
            self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
            #Elementos estructurantes para filtros morfologicos
            self.kernelOp = np.ones((3,3), np.uint8)
            #kernelOp2 = np.ones((5,5),np.uint8)
            self.kernelCl = np.ones((8, 8), np.uint8)
        
        elif self.process_type == self.PROCESS_DNN:
            # Inicia DNN
            prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
            model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        self.resolution = resolution
        self.frame = None

        if self.process_type == self.PROCESS_HOG:
            _target = self.process_hog
        elif self.process_type == self.PROCESS_MOG:
            _target = self.process_mog
        elif self.process_type == self.PROCESS_DNN:
            _target = self.process_dnn
        else:
            _target = self.process_mog

        self._target_stop = False
        t = Thread(target=_target, args=())
        t.start()
        #t.join()

    def target_stop(self):
        self._target_stop = True

    def process(self, frame):
        self.frame = frame
        return self.frame

    def process_hog(self):
        while True:
            if self._target_stop:
                return

            if self.frame is None:
                continue

            # detect people in the image
            (rects, weights) = self.hog.detectMultiScale(self.frame, winStride=(4,4), padding=(8,8), scale=1.05)
        
            if len(rects) > 0:
                log.debug("detected {} peoples".format(len(rects)))

                # draw bounding boxes
                for (x, y, w, h) in rects:
                    self.frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imwrite("{}/objects/{}-{}-{}.jpg".format(current_dir, datetime.datetime.now().isoformat(), 
                    "HOG", len(rects)), self.frame)



    def process_mog(self):
        while True:
            if self._target_stop:
                return
            
            if self.frame is None:
                continue

            self.areath_min = self.max_area
            self.areath_max = self.min_area

            mask = self.fgbg.apply(self.frame)
            try:
                mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
                #Opening (erode->dilate) para quitar ruido.
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOp)
                #Closing (dilate -> erode) para juntar regiones blancas.
                mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, self.kernelCl)

            except:
                pass

            else:
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.areath_min and area < self.areath_max:
                        log.debug("area {}".format(area))
                        M = cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        x,y,w,h = cv2.boundingRect(cnt)
                        
                        self.frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        cv2.imwrite("{}/objects/{}-{}-a{}.jpg".format(current_dir, datetime.datetime.now().isoformat(), 
                            "MOG", area), self.frame)

    def process_dnn(self):
        while True:
            if self._target_stop:
                return
            
            if self.frame is None:
                continue
        
            blob = cv2.dnn.blobFromImage(self.frame, 0.007843, self.resolution, 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            #log.debug("Positive detections %s", detections.shape[2])

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                if confidence >= 0.65:
                    idx = int(detections[0, 0, i, 1])

                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([*self.resolution, *self.resolution])
                    (startX, startY, endX, endY) = box.astype("int")
                    # Log detection
                    log.debug("{} detected at {},{}".format(self.DNN_CLASSES[idx], startX, startY))

                    image = cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0,255,0), 2)
                    cv2.imwrite("{}/objects/{}-{}-x{}-y{}.jpg".format(current_dir, datetime.datetime.now().isoformat(), 
                        self.DNN_CLASSES[idx], startX, startY), image)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False, help="Source input, default camera")
    ap.add_argument("-p", "--process", required=False, help="Processs with (HOG, MOG), default no process")
    ap.add_argument("-P", "--preview", action='store_true', default=False, help="Preview (require connect to X server")
    ap.add_argument("-f", "--framerate", default=30, help="Framerate, default 30")
    ap.add_argument("-r", "--rotation", default=0, help="Rotation, default 0")

    ap.add_argument("--minarea", default=5, help="Min area detection")
    ap.add_argument("--maxarea", default=50, help="Max area detection")

    args = vars(ap.parse_args())

    resolution = (320, 240) # 76800    

    process = args.get('process')
    preview = args.get('preview')
    framerate = int(args.get('framerate'))
    rotation = int(args.get('rotation'))

    total_area = resolution[0] * resolution[1] 

    min_area = total_area / int(args.get('minarea'))
    max_area = total_area / int(args.get('maxarea'))

    log.info("starting video file thread...")
    vs = VideoStream(pi_cam=True, resolution=resolution, framerate=framerate, rotation=rotation).start()
    if process:
        ip = ImageProcessor(resolution, process.lower(), min_area=min_area, max_area=max_area)

    # ImageShow threading
    #im = ImageShow().start()
    #out = cv2.VideoWriter("{}/objects/{}-{}-video.avi".format(current_dir, datetime.datetime.now().isoformat(), process), 
    #        cv2.VideoWriter_fourcc(*'XDIV'), 20.0, resolution)

    # allow the camera to warmup
    time.sleep(1)

    fps = FPS().start()

    while True:
        frame = vs.read()

        if process:
            # call to proccessing thread
            ip.process(frame)
            frame = ip.frame
        
        if preview:
            # Set frame to show
            #im.set_frame(frame)
            cv2.imshow("preview", frame)

        # save video
        #out.write(frame)

        fps.update()

        # if the `q` key was pressed, break from the loop
        if signal_handler.stop or cv2.waitKey(1) & 0xFF == ord('q'):
            # do a bit of cleanup
            if process:
                ip.target_stop()
            #im.stop()
            #out.release()
            vs.stop()
            cv2.destroyAllWindows()
            break

    # stop the timer and display FPS information
    fps.stop()
    log.info("elasped time: {:.2f}".format(fps.elapsed()))
    log.info("approx. FPS: {:.2f}".format(fps.fps()))
     
    # do a bit of cleanup
    if process:
        ip.target_stop()

    #im.stop()
    #out.release()
    vs.stop()
    cv2.destroyAllWindows()


class SignalHandler:
    stop = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.safe_stop)
        signal.signal(signal.SIGTERM, self.safe_stop)

    def safe_stop(self, signum, frame):
        log.debug("Safe stop process %s", signum)
        self.stop = True

signal_handler = SignalHandler()

if __name__ == '__main__':
    main()
