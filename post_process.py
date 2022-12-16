
# import packages
import cv2
import os
import numpy as np
import logging
import sys
import skimage.measure as sm
import matplotlib.pyplot as plt
import glob
from skimage.segmentation import clear_border
import re
from scipy import ndimage
from skimage import io


#########################################################################


### Parameters user sets: ###
# location of where predicted images are
predicted_img_folder = "/content/MARS_Net/models/results/predict_wholeframe_round1_demo_VGG19_dropout/C2C12_myoblast_training/frame3_A_repeat0/"

# name of model
model_name = "demo_VGG19_dropout"

# where to save images
save_folder = "/content/MARS_Net/post_processed/"

# type of image (ex. png, tif)
img_type = "png"

# file location to original images if you want to crop those
orig_img_folder = "/content/MARS_Net/assets/C2C12_myoblast_training/control_200_350/"


########################################################################


### Running via Terminal ###


# extra things to set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# sys.path.append('..')

logging.info('================= Reading and Processing Data =================')

# make save folder
os.makedirs(os.path.dirname(save_folder+model_name), exist_ok=True)
os.makedirs(os.path.dirname(save_folder+model_name+'/'+"postprocessed_images/"), exist_ok=True)
os.makedirs(os.path.dirname(save_folder+model_name+'/'+"original_images/"), exist_ok=True)
os.makedirs(os.path.dirname(save_folder+model_name+'/'+"overlay_images/"), exist_ok=True)


files = glob.glob(predicted_img_folder + "*."+img_type)
files_orig = glob.glob(orig_img_folder + "*."+img_type)


# sort by specific number
def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'-(00\d+)', string_)] ## NOTE: CHANGE REGEX BASED ON DATA

# sorted, final version
files_final = sorted(files, key = natural_key)
files_orig_final = sorted(files_orig, key = natural_key)

print(files_final)

# read in images
images = [cv2.imread(file) for file in files_final]
images_orig = [cv2.imread(file) for file in files_orig_final]



# print number of images to verify
print("Number of Predicted Images: " + str(len(images)))


img_num=0
# cycling through images
for i in range(0,len(images)):
    img_num +=1
    img = images[i]
    img_orig = images_orig[i]
    # need one original for overlay, the other to save
    img_original = img_orig.copy()
    print("Evaluating Image: " + str(i+1))
    # change to gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # img_gray = img

    # threshold image
    ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    clear = clear_border(thresh)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    
    result = np.zeros((labels.shape), np.uint8)
    # result image labelled
    for i in range(0, nlabels - 1):
        if areas[i] >= 300: #keep
            result[labels == i + 1] = 255

    # fill in holes
    result = ndimage.binary_fill_holes(result).astype(int)
    result = np.where(result==1, 255, 0)

    # for overlay
    contours, hierarchy = cv2.findContours(result.astype(np.uint8),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.drawContours(img_orig, contours, -1, (0,0,255), 3)

    # save images
    file_name = 'postp_image_{}.png'.format(str(img_num))
    cv2.imwrite(save_folder+model_name+'/'+"postprocessed_images/"+ file_name ,result)

    file_name_overlay = 'overlay_image_{}.png'.format(str(img_num))
    cv2.imwrite(save_folder+model_name+'/'+"overlay_images/"+ file_name_overlay, overlay)

    file_name_orig = 'original_image_{}.png'.format(str(img_num))
    cv2.imwrite(save_folder+model_name+'/'+"original_images/"+ file_name_orig ,img_original)
