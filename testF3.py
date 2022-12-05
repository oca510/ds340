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
from result_gen2 import load_model, preprocess_image, get_plate, segment, mask_gen, find_cand, find_rect, make_crop, recogn, output

def test(directory):
	# iterate over files in that directory
	wpod_net_path = "models/wpod-net.json"
	wpod_net = load_model(wpod_net_path)
	json_file = open('models/MobileNets_character_recognition5.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("models/License_character_recognition5.h5")
	print("[INFO] Model loaded successfully...")
	labels = LabelEncoder()
	labels.classes_ = np.load('models/license_character_classes5.npy')
	print("[INFO] Labels loaded successfully...")
	images = []
	for filename in os.listdir(directory):
		if filename[0] != ".":
			f = os.path.join(directory, filename)
			# checking if it is a file
			if os.path.isfile(f):
				images.append(f)
	#num_to_select = 50                      
	#images = random.sample(images, num_to_select)
	count = 0
	mistakes = {}
	for img in images:
		answer = img.replace('/',',').replace('.',',').split(',')[1]
		print("Start: ", img, answer)
		result = output(img, wpod_net, model, labels)
		if result == answer:
			count +=1
			print("Correct: ", result)
		else:
			if result == None:
				mistakes[answer] = result
			else: 
				if len(result) != len(answer):
					mistakes[answer] = result +  "lengh problem"
				else: 
					mistakes[answer] = result
			print("False: ", result)
	Accuracy = "Accuracy: {acc}%\n".format(acc = count/len(images) * 100)
	return Accuracy, mistakes

if __name__ == "__main__":
	#print(output("Plate_examples/india_car_plate.jpg"))
	directory = 'fusion_data_100'
	#directory = 'check3_2'
	Accuracy, mistakes = test(directory)
	print(mistakes)
	print(Accuracy)