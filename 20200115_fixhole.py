import numpy as np
import cv2 
import matplotlib.pyplot as plt

OUTPUT_PATH = "output/green_in_glass"
INPUT_PATH = "D:\\Datasets\\green_in_glass\\%03d%03d.png"

TOTAL_IMAGE_PER_CAMERA = 40
THESHOLD_RATIO = 0.10 # ratio [0.0 - 1.0]
TOTAL_CAMERA = 10
BACKGROUND_COLOR = [0, 0, 0]

def get_diff_mask(camerea_id,current_shot):
    previous_shot = (current_shot - 1) % TOTAL_IMAGE_PER_CAMERA
    # read image
    image_prev_uint = cv2.imread(INPUT_PATH % (camerea_id,previous_shot)) 
    image_current_uint = cv2.imread(INPUT_PATH % (camerea_id,current_shot)) 
    # convert to RGB
    image_prev_uint = cv2.cvtColor(image_prev_uint,cv2.COLOR_BGR2RGB)
    image_current_uint = cv2.cvtColor(image_current_uint,cv2.COLOR_BGR2RGB)
    # rotate
    image_prev_uint = cv2.rotate(image_prev_uint, cv2.ROTATE_90_CLOCKWISE)
    image_current_uint = cv2.rotate(image_current_uint, cv2.ROTATE_90_CLOCKWISE)
    # convert to [0-1]
    image_prev =  image_prev_uint / 255.0
    image_current = image_current_uint / 255.0
    # difference mask between 2 images
    diff_mask = np.linalg.norm(image_current - image_prev, axis=-1)
    diff_mask = (diff_mask > THESHOLD_RATIO) * 1.0
    #remove noise from sensor (hope this not ruin image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    denoised_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    return denoised_mask

for CAMERA_NUMBER in range(TOTAL_CAMERA):
    foreground_prob = get_diff_mask(CAMERA_NUMBER,0)
    for i in range(1,40):
        foreground_prob = cv2.bitwise_or(foreground_prob,get_diff_mask(CAMERA_NUMBER,i))
    image_flooded = (foreground_prob.copy() * 255.0).astype(np.uint8)
    image_height, image_width = image_flooded.shape[:2]
    flood_mask = np.zeros((image_height+2,image_width+2),dtype=np.uint8)
    # top bar
    for i in range(image_flooded.shape[1]):
        if image_flooded[0,i] != 255:
            cv2.floodFill(image_flooded, flood_mask, (0,i), 255)
    # left bar
    for i in range(image_flooded.shape[0]):
        if image_flooded[i,0] != 255:
            cv2.floodFill(image_flooded, flood_mask, (i,0), 255)

    # right bar
    most_right = image_flooded.shape[1] -1
    for i in range(image_flooded.shape[0]):
        if image_flooded[i,most_right] != 255:
            cv2.floodFill(image_flooded, flood_mask, (i,most_right), 255)

    # bottom bar 
    most_bottom = image_flooded.shape[0] -1
    for i in range(image_flooded.shape[1]):
        if image_flooded[most_bottom,i] != 255:
            cv2.floodFill(image_flooded, flood_mask, (most_bottom,i), 255)
    background_mask = flood_mask[1:-1,1:-1]

    # remove background of the set
    for IMAGE_NUMBER in range(TOTAL_IMAGE_PER_CAMERA):
        image = cv2.imread(INPUT_PATH % (CAMERA_NUMBER, IMAGE_NUMBER))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image[background_mask == 1] = BACKGROUND_COLOR
        cv2.imwrite("%s/cam%03d_%05d.png"%(OUTPUT_PATH,CAMERA_NUMBER,IMAGE_NUMBER), image)
