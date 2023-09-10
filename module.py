import cv2 
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
center = (blank.shape[1]//2, blank.shape[0]//2)


def display_video(location):
	video = cv2.VideoCapture(location)
	while True:
		isTrue, frame = video.read()
		cv2.imshow('Video', frame)

		if cv2.waitKey(20) and 0xff==ord('d'):
			break
def video_rescale(location):
	video = cv2.VideoCapture(location)
	while True:
		isTrue, frame = video.read()
		rescaleFrame = rescale(frame)
		cv2.imshow('Video', rescaleFrame)

		if cv2.waitKey(20) and 0xff==ord('d'):
			break
	video.release()
	cv2.destroyAllWindows()

def rescale(frame, scale=0.75):
	height = int(frame.shape[0]*scale)
	
	width = int(frame.shape[1]*scale)

	dimensions = (width, height)
	return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
	
def r(img, start_point, end_point, color, thickness):
	cv2.rectangle(img, start_point, end_point, color, thickness)
	cv2.imshow('Blank', img)

def c(image, center_coordinates, radius, color, thickness):
	cv2.circle(image, center_coordinates, radius, color, thickness)
	cv2.imshow('Circle',image)

def text(image, text, org, font, fontScale, color, thickness):
	cv2.putText(image, text, org, font, fontScale, color, thickness)
	cv2.imshow('Text', image)



#shows all the edges in an image
def canny(img):
	im = cv2.imread(img)
	canny = cv2.Canny(im, 125,175)
	cv2.imshow('Canny Edge', canny)
	
	return canny


#increases the thickness of the canny edge
def dilate(canny_img):
	
	dilate = cv2.dilate(canny_img, (3,3), iterations=1)
	cv2.imshow('Dilated image', dilate)
	return dilate




#Used to get back the canny image after being dilated
def erode(dilated_img):
	
	erode = cv2.erode(dilated_img, (3,3), iterations=1)
	cv2.imshow('Eroded image', erode)
	
	return erode

#deletes the rest of the image except the specified region
def cropped(img):
	im = cv2.imread(img)
	cropped = im[50:200, 200:400]
	cv2.imshow('Cropped_img', cropped)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


#shifting the image accros x and y axis
def transformation(im):
     img = cv2.imread(im, 0)
     rows, cols = img.shape
     M = np.float32([[1, 0, 100], [0, 1,50]])
     sheared_img  = cv2.warpAffine(img, M, (cols,rows))
     cv2.imshow('img', sheared_img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


#flips the image vertically
def flip_vertical(img):
     im = cv2.imread(img, 0)
     rows, cols = im.shape
     M = np.float32([[1, 0, 0], [0, -1, rows],[0, 0, 1]])
     sheared_img = cv2.warpPerspective(im, M, (cols,rows))
     cv2.imshow('Flipped Vertically', sheared_img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
#flips the image horizontally
def flip_horizontal(img):
	im = cv2.imread(img, 0)
	rows,cols = im.shape
	M=np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]])
	

	sheared_img = cv2.warpPerspective(im, M, (cols,rows))
	cv2.imshow('Flipped Horizontally', sheared_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#rotates an image
def rotation(img):
	im = cv2.imread(img,0)
	rows, cols = im.shape
	M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
	img_rotation = cv2.warpAffine(im, cv2.getRotationMatrix2D((cols/2, rows/2),30,0.6), (cols, rows))
	cv2.imshow('Rotated Img', img_rotation)
	cv2.imwrite('rotation_out.jpg', img_rotation)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def shearing_x_axis(img):
     im = cv2.imread(img,0)
     rows, cols = im.shape
     M = np.float32([[1,0.5,0], [0,1,0], [0,0,1]])
     sheared_img = cv2.warpPerspective(im, M, (int(cols*1.5), int(rows*1.5)))
     cv2.imshow('Sheared on X-axis', sheared_img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

def shearing_y_axis(img):
     im= cv2.imread(img, 0)
     rows, cols = im.shape
     M = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
     sheared_img = cv2.warpPerspective(im, M, (int(cols*1.5), int(rows*1.5)))
     cv2.imshow('Sheared on y-axis', sheared_img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

#BLURRING TECHNIQUES
#2D Convolution blurring involves applying a kernel r to an image to reduce  noise , resulting in a smoother and  version of the image.
def convolution2D(img):
	im = cv2.imread(img)
	kernel = np.ones((5,5), np.float32)/25
	blur = cv2.filter2D(im, -1, kernel)
	cv2.imshow('Original Image', im)
	cv2.imshow('Kernel Blur', blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Averaging involves replacing the centre pixel in the kernel window with the average value of its neighboring pixels.Produces most blur 
def averaging(img):
	im = cv2.imread(img)
	averageBlur = cv2.blur(im, (5,5))
	cv2.imshow('Original Image', im)
	cv2.imshow('Average Blur', averageBlur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Median blur is an image processing technique that replaces the centre pixel's value with the median value of its neighboring pixels 
def medianBlur(img):
	im = cv2.imread(img)
	median = cv2.medianBlur(im,9)
	cv2.imshow('Original Image', im)
	cv2.imshow('Median Blur', median)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#this function concerns more about the edges and smoothens the image by preserving the edges
def bilateralBlur(img):
	im = cv2.imread(img)
	#increasing sigma space and sigma colors will increase the range  of pixels and more colors in the surrounding pixels taken into consideration while taking average for the cetnre pixels respectively
	bilateral = cv2.bilateralFilter(im ,9, 75, 75)
	cv2.imshow('Original Image', im)
	cv2.imshow('Bilateral Blur', bilateral)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Gaussian blur t uses a Gaussian filter to reduce noise  by replacing each pixel's value with a weighted average(each pixel is assigned a specific weight or importance) of its neighboring pixels
def gaussianBlur(img):
	im = cv2.imread(img)
	cv2.imshow('Original Image', im)
	blur = cv2.GaussianBlur(im, (5,5), cv2.BORDER_DEFAULT)
	cv2.imshow('Gaussian Blur', blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def contour_detection(im):
     img = cv2.imread(im)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     edged = cv2.Canny(gray, 30,300)
     contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     cv2.imshow('canny edges after contouring', edged)
     print('number of contours ='+ str(len(contours)))
     cv2.drawContours(img, contours, -1, (0,255,0),2)
     cv2.imshow('contours', img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()



#shows the regions and intensities of  the color is in an image 
def gray_colorSpace_split(im):
     img = cv2.imread(im)
     B,G,R = cv2.split(img)
     cv2.imshow('Original', img)
     #if the grayscale image is darker it means that there is less of that color in that specific region and  if it is light there is more of that color in that specific region
     cv2.imshow('Blue', B)
     
     cv2.imshow('Green', G)
     
     cv2.imshow('Red', R)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


def BGR_split(im):
	
	img = cv2.imread(im)
	blank = np.zeros(img.shape[:2], dtype='uint8')
	#lighter the region is more the color intensity is present in th region if it is darker then it is less
	b,g,r = cv2.split(img)
	blue = cv2.merge([b, blank, blank])
	red = cv2.merge([blank, blank, r])
	green = cv2.merge([blank, g, blank])
	cv2.imshow('Red', red)
	cv2.imshow('Green', green)
	cv2.imshow('Blue', blue)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def merge_bgr(im):
	
     img = cv2.imread(im)
     cv2.imshow('normal photo', img)
     B,G,R = cv2.split(img)

     merged = cv2.merge([B,G,R])
     cv2.imshow("merged", merged)
     cv2.waitKey(0)
	

#places two images on top of eaach other and shows the intersection
def bitwise_and():
	blank = np.zeros((400,400), dtype='uint8')
	rectangle = cv2.rectangle(blank.copy(),(30,30),(370,370), 255, -1)

	circle = cv2.circle(blank.copy(),(200,200), 200, 255, -1  )
	cv2.imshow('rectangle', rectangle)
	cv2.imshow('circle', circle)
	b_and = cv2.bitwise_and(rectangle, circle)
	cv2.imshow('Bitwise AND', b_and)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#places two images on top of each other and returns both intersecting and non-intersecting regions
def bitwise_or():
	blank = np.zeros((400,400), dtype='uint8')
	rectangle = cv2.rectangle(blank.copy(),(30,30),(370,370), 255, -1)

	circle = cv2.circle(blank.copy(),(200,200), 200, 255, -1  )
	cv2.imshow('rectangle', rectangle)
	cv2.imshow('circle', circle)
	b_or = cv2.bitwise_or(rectangle, circle)
	cv2.imshow('Bitwise OR', b_or)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#places two images on top of each other and returns the non-intersecting regions
#if you subtract xor from or you get and
def bitwise_xor():
	blank = np.zeros((400,400), dtype='uint8')
	rectangle = cv2.rectangle(blank.copy(),(30,30),(370,370), 255, -1)
     
	circle = cv2.circle(blank.copy(),(200,200), 200, 255, -1  )
	cv2.imshow('rectangle', rectangle)
	cv2.imshow('circle', circle)
	b_xor = cv2.bitwise_xor(rectangle, circle)
	cv2.imshow('Bitwise XOR', b_xor)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#inverts the binary color
def bitwise_not():
	blank = np.zeros((400,400), dtype='uint8')
	rectangle = cv2.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
	circle = cv2.circle(blank.copy(),(200,200), 200, 255, -1  )
	cv2.imshow('rectangle', rectangle)
	cv2.imshow('circle', circle)
	not_rect = cv2.bitwise_not(rectangle)
	not_circle = cv2.bitwise_not(circle)
	cv2.imshow('Bitwise NOT on rectangle', not_rect)
	cv2.imshow('Bitwise NOT on circle', not_circle)
	cv2.waitKey(0)

#shows the region of interest
def masking(im):
     # masking allows us to focus on certain parts of an image
     img = cv2.imread(im)
     cv2.imshow('Photo', img)
     blank = np.zeros(img.shape[:2], dtype='uint8')
     # dimensions of mask must be of the same size of image
     mask = cv2.circle(blank, (img.shape[1]//2, img.shape[0]//2), 135,255,-1)
     cv2.imshow('Mask', mask)
     masked = cv2.bitwise_and(img, img, mask=mask)
     cv2.imshow('Masked', masked)
     cv2.waitKey(0)


def histogram_gray(img):
	im = cv2.imread(img)
	gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	gray_hist = cv2.calcHist([gray],[0], None, [256], [0,256] )
	plt.figure()
	plt.title('Grayscale Histogram')
	plt.xlabel('Bins')
	plt.ylabel('No. of pixels')
	plt.plot(gray_hist)
	plt.xlim([0,256])
	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def histogram_bgr(im):
	img = cv2.imread(im)
	cv2.imshow('Cats', img)
	blank = np.zeros(img.shape[:2], dtype='uint8')
	mask = cv2.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
	masked = cv2.bitwise_and(img, img, mask=mask)
	cv2.imshow('Mask',  masked)
	plt.figure()
	plt.title('Colour Histogram')
	plt.xlabel('Bins')
	plt.ylabel('# of pixels')
	colors = ('b','g','r')
	for i,col in enumerate(colors):
		hist = cv2.calcHist([img],[i],mask,[256],[0,256])
		plt.plot (hist, color=col)
		plt.xlim([0,256])
	plt.show()
	cv2.waitKey(0)
     

def line(image, start_point, end_point, color, thickness):
	cv2.line(image, start_point, end_point, color, thickness)
	cv2.imshow('Line', image)

