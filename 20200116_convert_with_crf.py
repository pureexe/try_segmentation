import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool

INPUT_PATH = 'D:/Datasets/penguinguy/%03d%03d.png'
OUTPUT_PATH = "output/penguinguy_crf"

TOTAL_CAMERA = 10
TOTAL_IMAGE_PER_CAMERA = 40
THRESHOLD_RATIO = 0.10

OPENNING_KERNEL_SIZE = (5,5)

THRESHOLD_STRENG = 35
THRESHOLD_USE_TRIANGLE = False

USE_CLOSING_FOREGROUND = True
CLOSING_KERNEL = (30,30)

MP_POOL_SIZE = 10

#CRF configure
N_LABEL = 2
MAX_ITER = 200
PROBABILITY_THRESHOLD_OF_BACKGROUND = 0.5

#BACKGROUND
USE_BLUR_BACKGROUND = True
BLUR_KERNEL = (21,21)
NEW_BACKGROUND_COLOR = (0,0,0) # range 0-255
USE_MEAN_BACKGROUND = True


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
    diff_mask = (diff_mask > THRESHOLD_RATIO) * 1.0
    #remove noise from sensor (hope this not ruin image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,OPENNING_KERNEL_SIZE)
    denoised_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    return denoised_mask

# require by CRF
def whiten(im):
  H, W, C = im.shape
  # Flatten
  whitened_im = im.reshape(-1, C).astype('float')
  # Whiten the feature
  whitened_im -= np.mean(whitened_im, 0)
  whitened_im /= np.sqrt(np.mean(whitened_im ** 2, 0))
  return whitened_im.reshape(H, W, C)

# require by CRF
def rbf_kernel(dist, sigma=3):
  return np.exp(- np.sum(dist ** 2, 2) / sigma)

def processed_camera(CAMERA_NUMBER):
    print("Forground Prob Cam:%02d" % (CAMERA_NUMBER, ))
    foreground_prob = get_diff_mask(CAMERA_NUMBER,0)
    for i in range(1,40):
        foreground_prob = cv2.bitwise_or(foreground_prob,get_diff_mask(CAMERA_NUMBER,i))

    if USE_CLOSING_FOREGROUND:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,CLOSING_KERNEL)
        mask_closed = cv2.morphologyEx(foreground_prob, cv2.MORPH_CLOSE, kernel)
    else:
        mask_closed = foreground_prob

    # find boundary (rectangle) of object 
    mask_y, mask_x = np.nonzero(mask_closed)
    min_x = np.min(mask_x)
    max_x = np.max(mask_x)
    min_y = np.min(mask_y)
    max_y = np.max(mask_y)

    #flood fill to remove hole
    print("Flooding Cam:%02d" % (CAMERA_NUMBER, ))
    image_flooded = (mask_closed.copy() * 255.0).astype(np.uint8)
    image_height, image_width = image_flooded.shape[:2]
    flood_mask = np.zeros((image_height+2,image_width+2),dtype=np.uint8)
    # top bar
    if min_y != 0:
        for i in range(image_flooded.shape[1]):
            if image_flooded[0,i] != 255:
                cv2.floodFill(image_flooded, flood_mask, (0,i), 255)
    # left bar
    if min_x != 0:
        for i in range(image_flooded.shape[0]):
            if image_flooded[i,0] != 255:
                cv2.floodFill(image_flooded, flood_mask, (i,0), 255)

    # right bar
    most_right = image_flooded.shape[1] -1
    if max_x != most_right:
        for i in range(image_flooded.shape[0]):
            if image_flooded[i,most_right] != 255:
                cv2.floodFill(image_flooded, flood_mask, (i,most_right), 255)

    # bottom bar 
    most_bottom = image_flooded.shape[0] -1
    if max_y != most_bottom:
        for i in range(image_flooded.shape[1]):
            if image_flooded[most_bottom,i] != 255:
                cv2.floodFill(image_flooded, flood_mask, (most_bottom,i), 255)

    # we get background from floodfill
    background_mask = flood_mask[1:-1,1:-1]
    
    for IMAGE_NUMBER in range(TOTAL_IMAGE_PER_CAMERA):
        print("working on Cam:%02d, Shot:%02d" % (CAMERA_NUMBER, IMAGE_NUMBER))
        image_current_uint = cv2.imread(INPUT_PATH % (CAMERA_NUMBER,IMAGE_NUMBER)) 
        image_current_uint = cv2.rotate(image_current_uint, cv2.ROTATE_90_CLOCKWISE)
        
        #Threshold for foreground
        image_gray = cv2.cvtColor(image_current_uint,cv2.COLOR_BGR2GRAY)
        if THRESHOLD_USE_TRIANGLE:
            ret2,object_threshold = cv2.threshold(image_gray,200,255,cv2.THRESH_TRIANGLE)
        else:
            ret2,object_threshold = cv2.threshold(image_gray,35,255,cv2.THRESH_BINARY)        


        feat = whiten(image_current_uint.copy())
        H, W = image_current_uint.shape[:2]

        # Create pairwise conditional probability with nearest 4 neighbors
        curr_feat = feat[1:H - 1, 1:W - 1]
        top, bottom, left, right = np.zeros((H, W)), np.zeros(
            (H, W)), np.zeros((H, W)), np.zeros((H, W))

        compatibility = np.array([[2, 0], [0, 2]])
        top[1:H - 1,    1:W - 1] = rbf_kernel(curr_feat - feat[0:H - 2, 1:W - 1])
        bottom[1:H - 1, 1:W - 1] = rbf_kernel(curr_feat - feat[2:H,     1:W - 1])
        left[1:H - 1,   1:W - 1] = rbf_kernel(curr_feat - feat[1:H - 1, 0:W - 2])
        right[1:H - 1,  1:W - 1] = rbf_kernel(curr_feat - feat[1:H - 1, 2:W])

        # Create unary potential
        logit = np.zeros((H, W, N_LABEL))
        unary = np.zeros((H, W, N_LABEL))

        # Set unary potential
        unary[:,:] = [0.5, 0.5]
        unary[background_mask == 1] = [0, 1]
        unary[object_threshold == 255] = [1, 0]

        # Updating CRF
        for curr_iter in range(MAX_ITER):
            score = np.exp(logit)
            prob = score / (np.sum(score, 2, keepdims=True) + 1e-6)

            # Pass message
            logit[1:H - 1, 1:W - 1] = (
                left[  1:H - 1, 1:W - 1, np.newaxis] * prob[1:H - 1, 0:W - 2] +
                right[ 1:H - 1, 1:W - 1, np.newaxis] * prob[1:H - 1, 2:W] +
                top[   1:H - 1, 1:W - 1, np.newaxis] * prob[0:H - 2, 1:W - 1] +
                bottom[1:H - 1, 1:W - 1, np.newaxis] * prob[2:H,     1:W - 1]
            ).dot(compatibility) + unary[1:H - 1, 1:W - 1]

        #calculate probability of foreground and background
        z = np.exp(logit[1:H - 1, 1:W - 1])
        p = z / np.sum(z, 2, keepdims=True)
        probability = p[:,:,0]

        #create output image
        output_image = image_current_uint.copy()
        m_h, m_w = output_image.shape[:2]
        probability = cv2.resize(probability, (m_w,m_h))
        output_image[probability <= PROBABILITY_THRESHOLD_OF_BACKGROUND] = NEW_BACKGROUND_COLOR
        cv2.imwrite("%s/cam%03d_%05d.png"%(OUTPUT_PATH,CAMERA_NUMBER,IMAGE_NUMBER), output_image)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    params = list(range(TOTAL_CAMERA))
    pool = Pool(MP_POOL_SIZE)  
    pool.map(processed_camera, params)  
