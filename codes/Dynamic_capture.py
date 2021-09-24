# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:51:12 2021

@author: Ares684
"""
import cv2
import numpy as np
 
#调用usb摄像头
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M","J","P","G"))
pre_frame = None
flag = True
first = True
#帧差法获取运动对象
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    if flag:
        height,width = frame.shape[:2]
        print(height,width)
        flag = False
    
    #高斯模糊
    gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
    # 如果没有背景图像就将当前帧当作背景图片
    if pre_frame is None:
        pre_frame = gray_lwpCV
    else:
        # absdiff帧差法
        img_delta = cv2.absdiff(pre_frame, gray_lwpCV)
        #threshold二值化
        ret,thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)
        # 膨胀图像
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.dilate(thresh,kernel,iterations=1)
        
        cv2.imshow('thresh',thresh)
        # findContours检测物体轮廓
        image,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lline = []
        rline = []
        tline = []
        bline = []
        totalarea = 0
        
        if contours:
            for cnt in contours:
                totalarea = totalarea + cv2.contourArea(cnt)
                leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
                rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
                topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
                lline.append(leftmost[0])
                rline.append(rightmost[0])
                tline.append(topmost[1])
                bline.append(bottommost[1])
            lline.sort()
            rline.sort()
            tline.sort()
            bline.sort()
            #获得左上角点与右下角点
            ltpoint = tuple([lline[0],tline[0]])
            rbpoint = tuple([rline[-1],bline[-1]])
            
            if first:
                #从左边进来
                if rline[-1] < (width/2):
                    direct = 0
                #从右边进来
                else:
                    direct = 1
                first = False
            #测试部分,可调参数(较近20为宜，较近10为宜)
            flag1 = (rline[-1] < width-20) and (direct == 1)
            flag2 = (lline[0] > 20) and (direct == 0)
            if (flag1 or flag2) and totalarea > 1000:
                # 设置窗口的初始位置
                x = lline[0]
                y = tline[0]
                w = rline[-1] - lline[0]
                h = bline[-1] - tline[0]
                track_window = (x, y, w, h)
                # 设置追踪的ROI窗口
                roi = frame[y:y+h, x:x+w]
                
                cv2.rectangle(frame, ltpoint, rbpoint, (0,255,0),3)
                cv2.imshow("frame",frame)
                break
            cv2.rectangle(frame, ltpoint, rbpoint, (0,255,0),3)
                
        cv2.imshow("frame",frame)
        pre_frame = gray_lwpCV
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break

cv2.imwrite('test.png',roi)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# 设置终止条件，可以是10次迭代，也可以至少移动1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 应用meanshift来获取新位置
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # 在图像上绘制
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('frame',img2)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
    else:
        break
        
#关闭
cap.release()        
cv2.destroyAllWindows()
