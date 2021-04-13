import cv2
import numpy as np

min_width=80 #80
min_height=80 #80
offset=6   #6
line_height=550 #550
match = []
car= 0

def cent(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilate, cv2.MORPH_CLOSE , kernel)
    closed = cv2.morphologyEx (dilated, cv2.MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, line_height), (1200, line_height), (255,127,0), 3) 

    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contours = (w >= min_width) and (h >= min_height)
        if not valid_contours:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centroid = cent(x, y, w, h)
        match.append(centroid)
        cv2.circle(frame1, centroid, 4, (0, 0,255), -1)

        for (x,y) in match:
            if y<(line_height+offset) and y>(line_height-offset):
                car+=1
                cv2.line(frame1, (25, line_height), (1200, line_height), (0,0,255), 3)  
                match.remove((x,y))
                print("car is detected: "+str(car))        
       
    cv2.putText(frame1, "Total Vehicles detected: "+str(car), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    cv2.imshow("Detected" , frame1)
    
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
