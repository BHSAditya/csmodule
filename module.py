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



def line(image, start_point, end_point, color, thickness):
	cv2.line(image, start_point, end_point, color, thickness)
	cv2.imshow('Line', image)

