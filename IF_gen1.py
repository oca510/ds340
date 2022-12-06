# ignore warning 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pywt
import cv2
import numpy as np
import IF1 as IF

def read_dir(directory):
	datal = []
	for filename in os.listdir(directory):
		old = os.path.join(directory, filename)
		if os.path.isfile(old):
			if [*filename][0] != '.':
				datal.append(old)
	return datal

if __name__ == "__main__":
	dir1 = "dataset_final_length_copy_blur_100"
	dir2 = "dataset_final_length_copy_blur_100 copy"
	output_dir = "fusion_data_100/"
	data1 = read_dir(dir1)
	data2 = read_dir(dir2)
	n = 0 
	while 0 < len(data1):
		image = IF.imageFusion(data1[n], data2[n])
		output_name = os.path.join(output_dir, data1[n].split('/')[1])
		cv2.imwrite(output_name, image)
		n += 1
		print("Successful", n)



