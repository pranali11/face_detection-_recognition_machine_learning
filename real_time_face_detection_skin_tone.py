import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    Hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    gray= cv2.cvtColor (img,cv2.COLOR_BGR2GRAY)
    th2= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    lower_red = np.array([0, 50,50])
    upper_red = np.array([30,255,255])
    mask = cv2.inRange(Hsv,lower_red, upper_red)
    mask= cv2.medianBlur(mask,5)
    mask= cv2.medianBlur(mask,5)
    res = cv2.bitwise_and(th2, th2,mask=mask)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(res,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    dilation_1 = cv2.dilate(dilation,kernel,iterations = 1)
    dilation_2 = cv2.dilate(dilation_1,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation_2, cv2.MORPH_CLOSE, kernel)
    closing_1 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    dilation_3 = cv2.dilate(closing_1,kernel,iterations = 1)
    dilation_4 = cv2.dilate(dilation_3,kernel,iterations = 1)
    dilation_5 = cv2.dilate(dilation_4,kernel,iterations = 1)
    dilation_6 = cv2.dilate(dilation_5,kernel,iterations = 1)
    dilation_7 = cv2.dilate(dilation_6,kernel,iterations = 1)
    dilation_8 = cv2.dilate(dilation_7,kernel,iterations = 1)
    closing_2 = cv2.morphologyEx(dilation_8, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing_2,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    
    im2,contours,hierarchy = cv2.findContours(erosion ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    lent = len(contours)

    
    largest = cv2.contourArea(contours[0])
    for i in range(lent):
        if (largest <= cv2.contourArea(contours[i])):
            largest = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            ratio= float(w)/h
            if(ratio <= (1.0) ):
                if(ratio >= 0.4):
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.imshow('image',img)

    if (largest == cv2.contourArea(contours[0])):
        
        x,y,w,h = cv2.boundingRect(contours[0])
        ratio= float(w)/h
        if(ratio <= (1.0) ):
            if(ratio >= 0.4):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('image', img)
    print (largest)
        
            
    k=cv2.waitKey(30) & 0xff
    if k== 27:
        break

cap.release()   

cv2.destroyAllWindows()
