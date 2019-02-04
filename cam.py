import cv2
import numpy as np
import picamera
import picamera.array
from tracker import Person

# Setup genral y opencv
w = 1024
h = 768
frameArea = h*w
areaTH = frameArea/250
print('Area Threshold', areaTH)

persons = []

#Substractor de fondo
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#Elementos estructurantes para filtros morfologicos
kernelOp = np.ones((3,3),np.uint8)
#kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (w, h)

        while True:
            camera.capture(stream, 'bgr', use_video_port=True)
            # stream.array now contains the image data in BGR order
            frame = stream.array
                    
            mask = fgbg.apply(frame)

            try:
                mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
                #Opening (erode->dilate) para quitar ruido.
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOp)
                #Closing (dilate -> erode) para juntar regiones blancas.
                mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
            except:
                pass
            

            for i in persons:
                i.age_one() #age every person one frame


            # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > areaTH:
                    #################
                    #   TRACKING    #
                    #################
                    
                    #Falta agregar condiciones para multipersonas, salidas y entradas de pantalla.
                    
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)

                    new = True

            
            cv2.imshow('frame', mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # reset the stream before the next capture
            stream.seek(0)
            stream.truncate()

        cv2.destroyAllWindows()
