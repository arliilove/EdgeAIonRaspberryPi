# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:31:36 2021

@author: Ares684
"""
import imutils
import cv2
import numpy as np

vs = cv2.VideoCapture(0)

colorLowerred = np.array([0,43,35])
colorUpperred = np.array([10,255,255])
#绿色或偏青色
colorLowergreen = np.array([35,43,35])
colorUppergreen = np.array([99,255,255])

#行进状态
status = False

while True:
    ret,frame = vs.read()
    if not ret:
        print('No Camera')
        break
    frame = cv2.flip(frame,1)
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, colorLowerred, colorUpperred)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("red_mask",mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circlearea = 3.1415 * radius * radius
        
        flag = area/circlearea

        if radius > 10 and flag > 0.8 and status == True:
            print("red")
            status = False
           
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            
        
    mask = cv2.inRange(hsv, colorLowergreen, colorUppergreen)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("green_mask",mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circlearea = 3.1415 * radius * radius
        
        flag = area/circlearea

        if radius > 10 and flag > 0.8 and status == False:
            print("green")
            status = True
            
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

print("cleanup")
vs.release()
cv2.destroyAllWindows()
