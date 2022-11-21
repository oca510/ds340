# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import functools

def load_model(path):
	try:
		path = splitext(path)[0]
		with open('%s.json' % path, 'r') as json_file:
			model_json = json_file.read()
		model = model_from_json(model_json, custom_objects={})
		model.load_weights('%s.h5' % path)
		print("Loading model successfully...")
		return model
	except Exception as e:
		print(e)

def preprocess_image(image_path,resize=False):
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img / 255
	if resize:
		img = cv2.resize(img, (224,224))
	return img

def get_plate(image_path, wpod_net, Dmax=608, Dmin = 256):
	
	vehicle = preprocess_image(image_path)
	ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
	side = int(ratio * Dmin)
	bound_dim = min(side, Dmax)
	_ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
	return vehicle, LpImg#, cor

def segment(LpImg, resize=False):
	
	if (len(LpImg)): #check if there is at least one license image
		# Scales, calculates absolute values, and converts the result to 8-bit.
		plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
		row, col = plate_image.shape[:2]
		bottom = plate_image[row-2:row, 0:col]
		mean = cv2.mean(bottom)[0]
		bordersize = 5
		plate_image = cv2.copyMakeBorder(
			plate_image,
			top=bordersize,
			bottom=bordersize,
			left=bordersize,
			right=bordersize,
			borderType=cv2.BORDER_CONSTANT,
			#value=[mean,mean,mean]
		)
		if resize:
			plate_image = cv2.resize(plate_image, (224,224))
		# convert to grayscale and blur the image
		gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
		'''if check_col(gray) == 'black':
			gray = cv2.bitwise_not(gray)'''
		blur = cv2.GaussianBlur(gray,(7,7),0)

		# Applied inversed thresh_binary 
		binary = cv2.threshold(gray, 180, 255,
							 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	return plate_image, binary # , gray, blur, kernel3, thre_mor

def mask_gen(plate_image, input_img):
	_, labels = cv2.connectedComponents(input_img)
	mask = np.zeros(input_img.shape, dtype="uint8")
	total_pixels = plate_image.shape[0] * plate_image.shape[1]
	lower = total_pixels // 70 # heuristic param, can be fine tuned if necessary
	upper = total_pixels // 20 # heuristic param, can be fine tuned if necessary
	for (i, label) in enumerate(np.unique(labels)):
		# If this is the background label, ignore it
		if label == 0:
			continue
		# Otherwise, construct the label mask to display only connected component
		# for the current label
		labelMask = np.zeros(input_img.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# If the number of pixels in the component is between lower bound and upper bound, 
		# add it to our mask
		if numPixels > lower and numPixels < upper:
			mask = cv2.add(mask, labelMask)
	return mask

def find_cand(plate_image, mask):
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	def compare(rect1, rect2):
		if abs(rect1[1] - rect2[1]) > 10:
			return rect1[1] - rect2[1]
		else:
			return rect1[0] - rect2[0]
	cand = []
	for boxes in boundingBoxes:
		x,y,w,h = boxes
		cand.append(boxes)
	cand = sorted(cand, key=lambda cand: cand[0])
	return cand

def find_rect(plate_image, cand):
	def req(rect, plate_image,):
		x,y,w,h = rect
		#avg_h = sum([h[3] for h in rects])/len(rects)
		plate_h = float(plate_image.shape[0])
		plate_w = float(plate_image.shape[1])
		aspectRatio = w / float(h)
		#solidity = cv2.contourArea(c) / float(w * h)
		heightRatio = h / plate_h
		#req = aspectRatio < 5 and 0.1 < heightRatio < 0.95 and w > 2
		weightRatio = w / plate_w
		#req = aspectRatio < 1 and 0.5 < heightRatio < 0.95 and w > 2 and weightRatio < 1/3
		yborder = y+h
		xborder = x+w
		#req = aspectRatio < 1 and 0.3 < heightRatio < 0.95 and w > 2 and weightRatio < 1/3 and y > 0.2*plate_h and yborder < 0.9 * plate_w and yborder < 0.9 * plate_w and x > 0.1 * plate_w
		req1 = aspectRatio < 1 and 0.3 < heightRatio < 0.95 and w > 3 and weightRatio < 1/3 
		return req1
	n = 0
	rects = []
	while n < len(cand):
		rect = cand[n]
		x,y,w,h = rect
		req1= req(rect, plate_image)
		if req1: 
			rects.append(rect)
		n +=1
	rects = sorted(rects, key=lambda rect: rect[0])
	return rects
	
def make_crop(plate, mask, rects):
	plate = plate.copy()
	TARGET_WIDTH = 30
	TARGET_HEIGHT = 60
	crop_characters = []
	for rect in rects:
		x,y,w,h = rect
		# Crop the character from the mask
		crop = mask[y:y+h, x:x+w]
		# Get the number of rows and columns for each cropped image
		# and calculate the padding to match the image input of pre-trained model
		rows = crop.shape[0]
		columns = crop.shape[1]
		#crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)     
		crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
		# Prepare data for pbrediction
		_, crop = cv2.threshold(crop, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		crop_characters.append(crop)
		# Show bounding box and prediction on image
		cv2.rectangle(plate, (x,y), (x+w,y+h), (0, 255, 0), 1)
	return plate, crop_characters


def recogn(crop_characters, model, labels):
	def predict_from_model(image,model,labels):
		image = cv2.resize(image,(80,80))
		image = np.stack((image,)*3, axis=-1)
		prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
		return prediction

	final_string = ''
	for i,character in enumerate(crop_characters):
		title = np.array2string(predict_from_model(character,model,labels))
		final_string+=title.strip("'[]")
	return final_string

def output(test_image_path, wpod_net, model, labels):
	#test_image_path = "Plate_examples/india_car_plate.jpg"
	try:
		vehicle, LpImg = get_plate(test_image_path, wpod_net)
		plate_image, binary= segment(LpImg)
		mask = mask_gen(plate_image, binary)
		cand = find_cand(plate_image, mask)
		if len(cand) > 0:
			rects = find_rect(plate_image, cand)
			#print('Cand: ', len(cand))
			if len(rects) >0:
				result_img, crop_characters = make_crop(plate_image, mask, rects)
				print('Rects: ', rects)
				if len(crop_characters) == 0:
					print(len(crop_characters))
					return None
				else:
					final_string = recogn(crop_characters, model, labels)
					return final_string
			else: return None
		else: return None
	except AssertionError as e:
		print(repr(e))
		return None
