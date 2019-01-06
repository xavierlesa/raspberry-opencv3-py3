#!/usr/bin/env python
import numpy as np
import cv2
from tracker import Person
import time
import argparse


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


def rescale_frame(frame, percent=75):
    """
    Reescale the frame in certain percent
    """
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def get_centroid(x, y, w, h):
    """
    Return the centroid of vector
    """

    dx = (w + x)/2
    dy = (h + y)/2
    return int(dx), int(dy)


def main(**kwargs):

    stepper = kwargs.get('stepper')
    confidence_ts = float(kwargs.get('confidence', 0.6))
    show_person_id = kwargs.get('showid')
    resize = int(kwargs.get('resize') or 75)
    output = kwargs.get('output')
    _input = kwargs.get("input", 0) or 0

    print("Source %s" % _input)
    
    cap = cv2.VideoCapture(_input)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(kwargs["prototxt"], kwargs["model"])
    
    # Substraction Maks
    #fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #kernel = np.ones((3,3), np.uint8)

    W, H = (None, None)
    writer = None


    fps = cap.get(5)


    persons = []
    #id = 0
    _last = 0
    
    iteration = 0

    frames = int(fps/6)

    print("Iterate over {} fps".format(frames))

    while(cap.isOpened()):
        iteration += 1
        frame = cap.read()[1]
        if resize:
            frame = rescale_frame(frame, percent=resize)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]


        if writer is None and output:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, fps, (W, H), True)

        # age every person one frame
        #for p in persons:
        #    p.age_one()


        if iteration % frames:
            continue 

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Log detection
        print("Iteration: {} Total shapes {} detected".format(iteration, detections.shape[2] ))

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_ts:

                idx = int(detections[0, 0, i, 1])
                
                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box

                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cx, cy = get_centroid(startX, startY, endX, endY)

                #if show_person_id:
                #    cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                #frame = cv2.rectangle(frame,(startX, startY),(endX, endY),(0,255,0),2)

                new_person = True

                for i, person in enumerate(persons):
                    person.age_one()
                    print("Person iter: {} // Person ID: {} // Age: {}".format(i, person.id, person.age))
                    # Look for x,y displacement
                    if abs(cx - person.getX()) <= (endX - startX) and abs(cy - person.getY()) <= (endY - startY):
                        new_person = False
                        person.updateCoords( (startX, startY), (endX, endY) )
                        break

                    if person.timedOut():
                        # remove persons
                        index = persons.index(person)
                        persons.pop(index)
                        del person

                if new_person:
                    #id += 1
                    person = Person( (startX, startY), (endX, endY), 10*frames )
                    persons.append(person)


        if _last != len(persons):
            _last = len(persons)
            print("Se detectaron {} personas".format(_last))

        #print("score: {} - {},{} {},{}".format(confidence, startX, startY, endX, endY))

        # Remove background from substraction mask
        #fgmask = fgbg.apply(frame)
        #mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        #tmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]

        # MorphologyEx layer
        # remove noise
        #mask = tmask
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        #cv2.imshow('EROSION', cv2.erode(frame, kernel, iterations=1))
        #cv2.imshow('DILATION', cv2.dilate(frame, kernel, iterations=1))
        #cv2.imshow('MORPH_OPEN', cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel))
        #cv2.imshow('MORPH_CLOSE', cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel))
        #cv2.imshow('MORPH_GRADIENT', cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel))
        #cv2.imshow('MORPH_TOPHAT', cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel))
        #cv2.imshow('MORPH_BLACKHAT', cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel))

        cv2.putText(frame, "Persons {}".format(_last), (20, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)

        cv2.putText(frame, "Persons {}".format(_last), (20, H - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0), 1, cv2.LINE_AA)


        if show_person_id:
            person = persons[0]
            cv2.putText(frame, "P {} A {}".format(str(person.id)[:4], person.age),
                    (person.v1[0], person.v1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, person.v1, person.v2, (0,255,0),2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        cv2.imshow('Frame', frame)
        #cv2.imshow('Mask', mask)

        # Espera "n" para continuar
        if stepper:
            cv2.waitKey()

        
        #preisonar ESC para salir
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break 

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == '__main__':

    # Arguemnts
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-r", "--resize", help="Resize to n%, default 75%")
    ap.add_argument("-s", "--showid", action='store_true', help="Show ID of tacked objects")
    ap.add_argument("-t", "--stepper", action='store_true', help="Step by step")
    ap.add_argument("-c", "--confidence", type=float, default=0.6, help="Confidence threshold, default 0.6")
    kwargs = vars(ap.parse_args())

    main(**kwargs)
