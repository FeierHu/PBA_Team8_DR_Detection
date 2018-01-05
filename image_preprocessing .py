# import the necessary packages
import cv2
import tqdm
import os

TRAIN_DIR = 'mypics' 
IMG_SIZE = 256

# load all images in a directory
for img in tqdm(os.listdir(TRAIN_DIR)):
	path = os.path.join(TRAIN_DIR, img)
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# grab the dimensions of the image and calculate the center
# of the image
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)
 
# cropping the image
	a = int(h/2) - int(IMG_SIZE/2)
	b = int(h/2) + int(IMG_SIZE/2)
	c = int(w/2) - int(IMG_SIZE/2)
	d = int(w/2) + int(IMG_SIZE/2)

	cropped = image[a:b, c:d]

# write the cropped image to disk in JPG format
	cv2.imwrite("processed_images/" + img, cropped)
