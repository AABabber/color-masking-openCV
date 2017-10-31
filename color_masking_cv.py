import cv2  # Import OpenCV 3 as cv2... ?
import numpy as np

# To capture a video, you need to create a VideoCapture object. Its argument can
# be either the device index or the name of a video file. Device index is just the
# number to specify which camera. Normally one camera will be connected, so we
# simply pass 0.
cap = cv2.VideoCapture(0)

while(1):
	_, img = cap.read()

	# Converting frame to HSV
    #
    # cvtColor converts an image from one color-space to another.
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Defining ranges for green and red colors via HSV
    #
    # The array elements specify the hue, saturation, and value, respectively.
    # We don't need to understand how they're defined, but for further information:
    # https://en.wikipedia.org/wiki/HSL_and_HSV
	green_lower=np.array([40, 100, 100], np.uint8)
	green_upper=np.array([80, 255, 255], np.uint8)
	red_lower=np.array([136, 87, 111], np.uint8)
	red_upper=np.array([180, 255, 255], np.uint8)

    # inRange checks if the array elements lie between the elements of the
    # two other arrays. In doing so, we threshold the HSV image to
    # get only the green/red colors.
	green=cv2.inRange(hsv, green_lower, green_upper)
	red=cv2.inRange(hsv, red_lower, red_upper)

	# Morphological transformation - dilation
    #
    # Morphological transformations are some simple operations based on
    # the image shape. It needs two inputs: 1) our original image and 2)
    # a "structuring element" or "kernel" which decides the nature of operation.
	kernal = np.ones((5 ,5), "uint8")
	green=cv2.dilate(green, kernal)

    # Bitwise-AND mask and original image
    #
    # TODO: Output the mask and result side-by-side to see what the bitwise-AND is doing.
	res2=cv2.bitwise_and(img, img, mask = green)


	# Tracking the Red Color
    #
    # There are three arguments in the findContours function, the first one is "source
    # image", the second is "contour retrieval mode", and the third is "contour
    # approximation method".
    #
    # NOTE: There's some inconsistency in the docs here.
    # The reference for cv2.findContours specifies that the return value is "void".
    #
    # contours is a Python list of all the contours in the image. Each individual contour
    # is a Numpy array of (x, y) coordinates of boundary points of the object.
	(_, contours, hierarchy)=cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
            # Constructing the bounding boxes to be drawn around the detected red area
			x,y,w,h = cv2.boundingRect(contour)
			img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			cv2.putText(img, "RED color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

	# Tracking green
	(_, contours, hierarchy)=cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
            # Constructing the bounding boxes to be drawn around the detected green area
			x, y, w, h = cv2.boundingRect(contour)
			img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			cv2.putText(img, "Green detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # Constructing the masks for the red and green regions
	mask = cv2.inRange(hsv, red_lower, red_upper)
	mask2 = cv2.inRange(hsv, green_lower, green_upper)
	total_mask = cv2.bitwise_or(mask, mask2)

    # Leaving only the red/green regions in the result window
	output = cv2.bitwise_and(hsv, hsv, mask=total_mask)
	cv2.namedWindow("grey_feed", cv2.WINDOW_NORMAL)
	OutputImg = np.hstack([img, output])
	OutputImg = cv2.resize(OutputImg, (1100, 434))
	cv2.imshow("grey_feed", OutputImg)
	# cv2.imshow("hello", img)

    # The waitKey function in OpenCV is used to introduce a delay of n milliseconds
    # while rendering images to windows. It would hang forever and NOT go to the
    # next iteration if we were using cv2.waitKey(0).
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break
