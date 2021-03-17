import numpy as np
import cv2


film = cv2.VideoCapture('C:\Users\MYM\Desktop\clear.mp4')
bgKNN = cv2.createBackgroundSubtractorKNN ()
while (True):
    suc, frame = film.read ()
    if suc == True:
        height, width = frame.shape [ : 2]
        resize = cv2.resize (frame, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
        gry = cv2.cvtColor (resize, cv2.COLOR_BGR2GRAY)
        CLAHE_default = cv2.createCLAHE ()
        CLAHE_equ_def = CLAHE_default.apply(gry)
        bgr = cv2.cvtColor (CLAHE_equ_def, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor (bgr, cv2.COLOR_BGR2HSV)
        blr = cv2.bilateralFilter (resize, 3, 30, 30)
        if cv2.waitKey(1) == 27:
            break
        maskKNN = bgKNN.apply (blr)
        resKNN = cv2.bitwise_and (blr, blr, mask = maskKNN)
        maskKNN = cv2.merge ([cv2.merge ([maskKNN, maskKNN]), maskKNN])
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #gray = cv2.cvtColor (maskKNN, cv2.COLOR_BGR2GRAY)
        #_ , thresh = cv2.threshold (gray, 100, 255, cv2.THRESH_BINARY)
        #_ , labels = cv2.connectedComponents (thresh, connectivity = 8, ltype = cv2.CV_32S)
        #mask = np.zeros (thresh.shape, np.uint)
        #center = []
        #for label in np.unique (labels):
         #if label == 0:
             #continue
         #labelMask = np.zeros (thresh.shape, dtype = "uint8")
         #labelMask [labels == label] = 255
         #numPixels = cv2.countNonZero (labelMask)
         #if numPixels > 10:
           #M = cv2.moments (labelMask)
           #cX = int (M ['m10'] / M ['m00'])
           #cY = int (M ['m01'] / M ['m00'])
           #center.append ((cX, cY))
           #mask = cv2.add (mask, labelMask)
        
        #cv2.imshow ("Original frame", frame)
        cv2.imshow ('KNN background sibtractor', np.hstack ([blr, maskKNN, resKNN]))
        cv2.moveWindow ("KNN background sibtractor", 0, 0)
        cv2.resizeWindow ("KNN background sibtractor", 2000, 1000)
    else:
        break
film.release ()

cv2.destroyAllWindows ()
