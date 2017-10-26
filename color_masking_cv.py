import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
	_, img = cap.read()

	#converting frame to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#defining range of green
	green_lower=np.array([40, 100, 100], np.uint8)
	green_upper=np.array([80, 255, 255], np.uint8)

	red_lower=np.array([136, 87, 111], np.uint8)
	red_upper=np.array([180, 255, 255], np.uint8)

	green=cv2.inRange(hsv, green_lower, green_upper)
	red=cv2.inRange(hsv, red_lower, red_upper)

	#Morphological transformation, Dilation
	kernal = np.ones((5 ,5), "uint8")

	green=cv2.dilate(green,kernal)
	res2=cv2.bitwise_and(img, img, mask = green)


	#Tracking the Red Color
	(_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):

			x,y,w,h = cv2.boundingRect(contour)
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

	#Tracking green
	(_,contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>300):
			x,y,w,h = cv2.boundingRect(contour)
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(img,"Green detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

	mask = cv2.inRange(hsv, red_lower, red_upper)
	mask2 = cv2.inRange(hsv, green_lower, green_upper)
	total_mask = cv2.bitwise_or(mask,mask2)
	output = cv2.bitwise_and(hsv, hsv, mask = total_mask)
	cv2.namedWindow("grey_feed", cv2.WINDOW_NORMAL)
	OutputImg = np.hstack([img, output])
	OutputImg = cv2.resize(OutputImg, (1100,434))
	cv2.imshow("grey_feed", OutputImg)
	#cv2.imshow("hello", img)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break
