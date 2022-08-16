#This script was used for batch analysis of a set of more than 2000 photographs of Arabidopsis seedlings on agar plates
#This code is included just reference purposes only - Because of the large file sizes the folder containing all of these images and data output file are not included in this repository so it will not run. 
#It is also specific to our image set and output file so it is not intended for general use

import os
import argparse
from plantcv import plantcv as pcv
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import cluster_jordan 
import show_objects
import math

# Parse command-line arguments

def options():
    parser = argparse.ArgumentParser(description="Imaging processing for A. thaliana seedlings with PlantCV")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False, default="./temp")
    parser.add_argument("-r", "--outfile", help="Result file.", required=True)
    parser.add_argument("-w", "--writeimg", help="Write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug",
                        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    parser.add_argument("-k","--key", help="path to csv file linking images to plates")
    parser.add_argument("-d","--data", help="path to csv file with all data")
    args = parser.parse_args()
    return args

# the main workflow for leaf area on plates
def main():

    args = options()
    pcv.params.debug = args.debug

    # requires a key file which links each image name to the right plate
    # and experiment. Key file format .csv
    # example line: "IMG_0123.JPG,1,A" [image, plate, experiment]
    map_dict = {}
    with open(args.key) as map:
        map = csv.reader(map, delimiter=",")
        for img, plate, exp in map:
            img = img[-12:]
            map_dict[f"{img}"] = (plate, exp)
    
    # the image being processed
    imagelink = args.image
    
    # define the dictionary that will hold the total json entry for the plate
    area_dict = {}
    photo = imagelink[-12:]

    # what pla
    area_dict[f'plate'] = map_dict[f'{photo.upper()}']
    # label ex. "exp_A_plate_1_IMG_0123"
    label = f"exp_{area_dict['plate'][1]}_plate_{area_dict['plate'][0]}_{photo[0:8]}"
    print(label)

    # read in the img
    image = cv2.imread(f"{imagelink}")
    shape = np.shape(image)

    # make sure img is landscape
    if shape[0] > shape[1]:
        image = pcv.rotate(image, 90, crop = None)
    
    # remove below line if not for calibration
    # image = image[0:,:-400]

    # thresholding to center the image around the plate
    thresh = pcv.rgb2gray_hsv(rgb_img=image, channel="h")
    thresh = pcv.gaussian_blur(img=thresh, ksize=(101, 101), sigma_x=0, sigma_y=None)
    thresh = pcv.threshold.binary(gray_img=thresh, threshold=80, max_value=235, object_type="light")
    fill = pcv.fill(bin_img=thresh, size=350000)
    dilate = pcv.dilate(gray_img=fill, ksize=120, i=1)
    id_objects, obj_hierarchy = pcv.find_objects(img=image, mask=dilate)

    # crop image around plate
    cnt = id_objects[0]
    x,y,w,h = cv2.boundingRect(cnt)
    #img = image[(y):(y+h+30),(x+50):(x+w-90)]
    img = image[(y):(y+h),(x):(x+w)]

    # finding the minimum area rectange that surrounds the plate
    # it can be angled differently than the original image
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # if the angle of the top line of the min rectangle
    # is less than 14 degrees (correct detection)
    # we rotate the whole plate so that it is straightened
    angle = rect[2]
    if angle < 14.0 and angle > -14.0:
        center = rect[0]
        width, height = rect[1]
        height = height-20
        M = cv2.getRotationMatrix2D(center,angle,1.0)
        img = cv2.warpAffine(img, M, (int(width-100), int(height-50)))
    else: 
        width = w-100
        height = h-70

    img = img[0:, 100:-50]

    # begin the workflow for leaf area measurement
    # applying a blur reduces the impact of noise
    blur = pcv.gaussian_blur(img=img, ksize=(21, 21), sigma_x=0, sigma_y=None)
    b = pcv.rgb2gray_lab(rgb_img=blur, channel="b")
    # checking if the image is overexposed
    avg = np.average(img)
    std = np.std(img)
    if avg > 200 and std < 35:
        # hist_equalization helps differentiate pixels when the
        # image is improperly exposed
        b = pcv.hist_equalization(b)
        # t is the threshhold to be used later. equalization changes the threshold
        t = 252
    else: 
        t = 140
    # defining a threshold between the leaf and the background
    b_thresh = pcv.threshold.binary(gray_img=b, threshold= t-3, max_value=255, object_type="light")
    # filling in small gaps within each leaf
    bsa_fill1 = pcv.fill(bin_img=b_thresh, size=200)
    bsa_fill1 = pcv.closing(gray_img=bsa_fill1)
    bsa_fill1 = pcv.erode(gray_img = bsa_fill1, ksize = 3, i = 1)
    bsa_fill1 = pcv.dilate(gray_img=bsa_fill1, ksize = 3, i = 1)
    bsa_fill1 = pcv.fill(bin_img=bsa_fill1, size=200)
    # finding the leaves, which are white objects in the above binary mask
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=bsa_fill1)

    # defining ROI
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=img, x=100, y=300, h=1300, w=int(width-400))

    # list of objs, hierarchies say object or hole w/i object
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, 
    roi_contour=roi_contour, roi_hierarchy=roi_hierarchy,
    object_contour=id_objects, obj_hierarchy=obj_hierarchy, roi_type="partial")

    # clustering defined leaves into individual plants using predefined rows/cols
    clusters_i, contours, hierarchies = cluster_jordan.cluster_contours(img=img, roi_objects=roi_objects, 
    roi_obj_hierarchy=hierarchy, nrow=1, ncol=6, show_grid=True)

    # split the clusters into individual images for analysis
    output_path, imgs, masks = cluster_jordan.cluster_contour_splitimg(rgb_img=img, 
    grouped_contour_indexes=clusters_i, contours=contours, hierarchy=hierarchies,
     outdir=args.outdir)

    # cycling through the split images and acquiring area data
    # this is pretty specific to our own dataset
    # and depends on an edit i made to the original pcv object_composition function
    leaf_error = False
    num_plants = 0
    for i in range(0,6):
        pos = 7-(i+1)
        if clusters_i[i][0] != None:
            id_objects, obj_hierarchy = pcv.find_objects(img=imgs[num_plants], mask=masks[num_plants])
            obj, mask1 = pcv.object_composition(img=imgs[num_plants], contours=id_objects, hierarchy=obj_hierarchy)
            m = cv2.moments(obj)
            # moment 00 is the area (count of pixel intensities in binary img = area of white pixels)
            area = m['m00']
            num_plants += 1
            # leaves tend to be roughly circular or rounded
            center, expect_r = cv2.minEnclosingCircle(obj)
            r = math.sqrt(area/math.pi)
            # check to see if the leaf area takes up enough of the area of the minimum enclosing circle
            # usually, if it does not, suggests a speck of root/glare is included in the detection
            if r <= 0.35*expect_r:
                leaf_error = True
                print(f"warning: there may be an error detecting leaf {pos}")
            with open(args.data) as dataab:
                opendata = csv.reader(dataab, delimiter=',') 
                for exp, plate, tube, loc, CS_number, nsource, nconc in opendata:
                    if plate == area_dict['plate'][0] and exp == area_dict['plate'][1]:
                        loc = int(loc)
                        if pos == loc:
                            entry = {'position':pos, 'tube':tube, 'area':area, 'suspicious':leaf_error}
                            area_dict[f'plant_{pos}'] = entry
                            break
                        else:
                            entry = {'position':pos, 'tube':None, 'area':area, 'suspicious':leaf_error}
                        area_dict[f'plant_{pos}'] = entry
        else:
            with open(args.data) as dataab:
                opendata = csv.reader(dataab, delimiter=',') 
                for exp, plate, tube, loc, CS_number, nsource, nconc in opendata:
                    if plate == area_dict['plate'][0] and exp == area_dict['plate'][1]:
                        loc = int(loc)
                        if pos == loc:
                            entry = {'position':pos, 'tube':tube, 'area':0, 'suspicious':None}
                            area_dict[f'plant_{pos}'] = entry
                            break
                        else:
                            entry = {'position':pos, 'tube':None, 'area':0, 'suspicious':None}
                        area_dict[f'plant_{pos}'] = entry

    # this is the workflow for IDing the scale
    scale_crop=img
    # ROI
    roi_contour_scale, roi_hierarchy_scale = pcv.roi.rectangle(img=scale_crop, x=125, y=height-1000, h=780, w=width-540)

    # thresholding
    a_scale = pcv.rgb2gray_lab(rgb_img=scale_crop, channel="b")
    a_scale = pcv.hist_equalization(a_scale)
    a_scale = pcv.gaussian_blur(img=a_scale, ksize=(21, 21), sigma_x=0, sigma_y=None)
    a_scale_thresh = pcv.threshold.binary(gray_img=a_scale, threshold=253, max_value=255, object_type="light")
    if len(np.unique(a_scale_thresh)) != 1:
        a_scale_thresh = pcv.closing(gray_img=a_scale_thresh)
        if len(np.unique(a_scale_thresh)) != 1:
            a_scale_thresh = pcv.fill(bin_img = a_scale_thresh, size= 11000)
            if len(np.unique(a_scale_thresh)) != 1:
                a_scale_thresh = pcv.fill_holes(bin_img=a_scale_thresh)
                if len(np.unique(a_scale_thresh)) != 1:
                    a_scale_thresh = pcv.erode(gray_img=a_scale_thresh, ksize = 9, i = 1)
                    if len(np.unique(a_scale_thresh)) != 1:
                        a_scale_thresh = pcv.fill(bin_img=a_scale_thresh, size=500)
                        if len(np.unique(a_scale_thresh)) != 1:
                            a_scale_thresh = pcv.dilate(gray_img=a_scale_thresh, ksize = 9, i = 1)
    id_scale, obj_hierarchy = pcv.find_objects(img=scale_crop, mask=a_scale_thresh)
    roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                                roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                        object_contour=id_scale, obj_hierarchy=obj_hierarchy, roi_type="partial")
    # check if the method found a circular object of the correct size
    # if it did, then we stop here
    # if it did not, we try a different one
    # necesary in our case because the dots are different colors
    count = 0
    area_dict['scale'] = 0
    for object in roi_scale:
        m = cv2.moments(object)
        area = m['m00']
        (x,y), expect_r = cv2.minEnclosingCircle(object)
        r = math.sqrt(area/math.pi)
        if r >= 0.95*expect_r and area > 8000:
            id_scale = object
            hier = scale_hierarchy[0][count]
            area_dict[f'scale'] = area
            print(area)
            break
        count += 1

    if area_dict['scale'] == 0:
        a_scale_thresh = pcv.threshold.binary(gray_img=a_scale, threshold=249, max_value=255, object_type="light")
        if len(np.unique(a_scale_thresh)) != 1:
            a_scale_thresh = pcv.closing(gray_img=a_scale_thresh)
            if len(np.unique(a_scale_thresh)) != 1:
                a_scale_thresh = pcv.fill(bin_img = a_scale_thresh, size= 11000)
                if len(np.unique(a_scale_thresh)) != 1:
                    a_scale_thresh = pcv.fill_holes(bin_img=a_scale_thresh)
                    if len(np.unique(a_scale_thresh)) != 1:
                        a_scale_thresh = pcv.erode(gray_img=a_scale_thresh, ksize = 9, i = 1)
                        if len(np.unique(a_scale_thresh)) != 1:
                            a_scale_thresh = pcv.fill(bin_img=a_scale_thresh, size=500)
                            if len(np.unique(a_scale_thresh)) != 1:
                                a_scale_thresh = pcv.dilate(gray_img=a_scale_thresh, ksize = 9, i = 1)
                
    id_scale, obj_hierarchy = pcv.find_objects(img=scale_crop, mask=a_scale_thresh)
    roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                              roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                     object_contour=id_scale, obj_hierarchy=obj_hierarchy, roi_type="partial")

    count = 0
    for object in roi_scale: 
        m = cv2.moments(object)
        area = m['m00']
        # check if the shape is roughly a circle
        (x,y), expect_r = cv2.minEnclosingCircle(object)
        r = math.sqrt(area/math.pi)
        if r >= 0.95*expect_r and area > 8000:
            id_scale = object
            hier = scale_hierarchy[0][count]
            area_dict[f'scale'] = area
            break
        count += 1

    if area_dict['scale'] == 0:
        h_scale = pcv.rgb2gray_hsv(rgb_img=scale_crop, channel="v")
        h_scale = pcv.hist_equalization(h_scale)
        h_scale = pcv.gaussian_blur(img=h_scale, ksize=(31, 31), sigma_x=0, sigma_y=None)
        h_scale_thresh = pcv.threshold.binary(gray_img=h_scale, threshold=251, max_value=255, object_type="light")
        if len(np.unique(h_scale_thresh)) != 1:
            h_scale_thresh = pcv.closing(gray_img=h_scale_thresh)
            if len(np.unique(h_scale_thresh)) != 1:
                h_scale_thresh = pcv.fill(bin_img = h_scale_thresh, size= 12000)
                if len(np.unique(h_scale_thresh)) != 1:
                    h_scale_thresh = pcv.fill_holes(bin_img=h_scale_thresh)
                    if len(np.unique(h_scale_thresh)) != 1:
                        h_scale_thresh = pcv.erode(gray_img = h_scale_thresh, ksize = 9, i = 1)
                        if len(np.unique(h_scale_thresh)) != 1:
                            h_scale_thresh = pcv.fill(bin_img= h_scale_thresh, size=500)
                            if len(np.unique(h_scale_thresh)) != 1:
                                h_scale_thresh = pcv.dilate(gray_img= h_scale_thresh, ksize = 9, i = 1)
        id_scale_2, obj_hierarchy_2 = pcv.find_objects(img=scale_crop, mask=h_scale_thresh)
        roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                                roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                object_contour=id_scale_2, obj_hierarchy=obj_hierarchy_2, roi_type="partial")
        count2 = 0
        for object in roi_scale: 
            m = cv2.moments(object)
            area = m['m00']
            (x,y), expect_r = cv2.minEnclosingCircle(object)
            r = math.sqrt(area/math.pi)
            if r >= 0.95*expect_r and area > 8000: 
                id_scale = object
                hier = scale_hierarchy[0][count2]
                area_dict[f'scale'] = area
                print(area)
                break
            count2 += 1
    
    if area_dict['scale'] == 0:
        h_scale = pcv.rgb2gray_hsv(rgb_img=scale_crop, channel="v")
        h_scale = pcv.hist_equalization(h_scale)
        h_scale = pcv.gaussian_blur(img=h_scale, ksize=(31, 31), sigma_x=0, sigma_y=None)
        h_scale_thresh = pcv.threshold.binary(gray_img=h_scale, threshold=235, max_value=255, object_type="light")
        if len(np.unique(h_scale_thresh)) != 1:
            h_scale_thresh = pcv.closing(gray_img=h_scale_thresh)
            if len(np.unique(h_scale_thresh)) != 1:
                h_scale_thresh = pcv.fill(bin_img = h_scale_thresh, size= 12000)
                if len(np.unique(h_scale_thresh)) != 1:
                    h_scale_thresh = pcv.fill_holes(bin_img=h_scale_thresh)
                    if len(np.unique(h_scale_thresh)) != 1:
                        h_scale_thresh = pcv.erode(gray_img = h_scale_thresh, ksize = 9, i = 1)
                        if len(np.unique(h_scale_thresh)) != 1:
                            h_scale_thresh = pcv.fill(bin_img= h_scale_thresh, size=500)
                            if len(np.unique(h_scale_thresh)) != 1:
                                h_scale_thresh = pcv.dilate(gray_img= h_scale_thresh, ksize = 9, i = 1)
        id_scale_2, obj_hierarchy_2 = pcv.find_objects(img=scale_crop, mask=h_scale_thresh)
        roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                                roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                object_contour=id_scale_2, obj_hierarchy=obj_hierarchy_2, roi_type="partial")
        count2 = 0
        for object in roi_scale: 
            m = cv2.moments(object)
            area = m['m00']
            (x,y), expect_r = cv2.minEnclosingCircle(object)
            r = math.sqrt(area/math.pi)
            if r >= 0.95*expect_r and area > 8000: 
                id_scale = object
                hier = scale_hierarchy[0][count2]
                area_dict[f'scale'] = area
                print(area)
                break
            count2 += 1

    if area_dict['scale'] == 0:
        h_scale = pcv.rgb2gray_hsv(rgb_img=scale_crop, channel="v")
        h_scale = pcv.hist_equalization(h_scale)
        h_scale = pcv.gaussian_blur(img=h_scale, ksize=(31, 31), sigma_x=0, sigma_y=None)
        h_scale_thresh = pcv.threshold.binary(gray_img=h_scale, threshold=245, max_value=255, object_type="light")
        if len(np.unique(h_scale_thresh)) != 1:
            h_scale_thresh = pcv.closing(gray_img=h_scale_thresh)
            if len(np.unique(h_scale_thresh)) != 1:
                h_scale_thresh = pcv.fill(bin_img = h_scale_thresh, size= 12000)
                if len(np.unique(h_scale_thresh)) != 1:
                    h_scale_thresh = pcv.fill_holes(bin_img=h_scale_thresh)
                    if len(np.unique(h_scale_thresh)) != 1:
                        h_scale_thresh = pcv.erode(gray_img = h_scale_thresh, ksize = 9, i = 1)
                        if len(np.unique(h_scale_thresh)) != 1:
                            h_scale_thresh = pcv.fill(bin_img= h_scale_thresh, size=500)
                            if len(np.unique(h_scale_thresh)) != 1:
                                h_scale_thresh = pcv.dilate(gray_img= h_scale_thresh, ksize = 9, i = 1)
        id_scale_2, obj_hierarchy_2 = pcv.find_objects(img=scale_crop, mask=h_scale_thresh)
        roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                                roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                object_contour=id_scale_2, obj_hierarchy=obj_hierarchy_2, roi_type="partial")
        count2 = 0
        for object in roi_scale: 
            m = cv2.moments(object)
            area = m['m00']
            (x,y), expect_r = cv2.minEnclosingCircle(object)
            r = math.sqrt(area/math.pi)
            if r >= 0.95*expect_r and area > 8000: 
                id_scale = object
                hier = scale_hierarchy[0][count2]
                area_dict[f'scale'] = area
                print(area)
                break
            count2 += 1

    if area_dict['scale'] == 0:
        h_scale = pcv.rgb2gray_hsv(rgb_img=scale_crop, channel="h")
        h_scale = pcv.hist_equalization(h_scale)
        h_scale = pcv.gaussian_blur(img=h_scale, ksize=(31, 31), sigma_x=0, sigma_y=None)
        h_scale_thresh = pcv.threshold.binary(gray_img=h_scale, threshold= 10, max_value=255, object_type="dark")
        if len(np.unique(h_scale_thresh)) != 1:
            h_scale_thresh = pcv.closing(gray_img=h_scale_thresh)
            if len(np.unique(h_scale_thresh)) != 1:
                h_scale_thresh = pcv.fill(bin_img = h_scale_thresh, size= 12000)
                if len(np.unique(h_scale_thresh)) != 1:
                    h_scale_thresh = pcv.fill_holes(bin_img=h_scale_thresh)
                    if len(np.unique(h_scale_thresh)) != 1:
                        h_scale_thresh = pcv.erode(gray_img = h_scale_thresh, ksize = 9, i = 1)
                        if len(np.unique(h_scale_thresh)) != 1:
                            h_scale_thresh = pcv.fill(bin_img= h_scale_thresh, size=500)
                            if len(np.unique(h_scale_thresh)) != 1:
                                h_scale_thresh = pcv.dilate(gray_img= h_scale_thresh, ksize = 9, i = 1)
        id_scale_2, obj_hierarchy_2 = pcv.find_objects(img=scale_crop, mask=h_scale_thresh)
        roi_scale, scale_hierarchy, scale_mask, scale_area = pcv.roi_objects(img=scale_crop, 
                                                                roi_contour=roi_contour_scale, roi_hierarchy=roi_hierarchy_scale,
                                                                object_contour=id_scale_2, obj_hierarchy=obj_hierarchy_2, roi_type="partial")
        count2 = 0
        for object in roi_scale: 
            m = cv2.moments(object)
            area = m['m00']
            (x,y), expect_r = cv2.minEnclosingCircle(object)
            r = math.sqrt(area/math.pi)
            if r >= 0.95*expect_r and area > 8000: 
                id_scale = object
                hier = scale_hierarchy[0][count2]
                area_dict[f'scale'] = area
                print(area)
                break
            count2 += 1

    # creating the json entry for the image
    areas = {}
    areas[f'{photo}'] = area_dict
    # if there is an error, it does not add the data to the file
    # remove this check depending on needs
    if area_dict["scale"] != 0 and leaf_error != True:
        # creates json file if one does not already exist
        dict = {}
        if not os.path.isfile(f'{args.outfile}.json'):
            with open(f'{args.outfile}.json', 'w') as file:
                json.dump(dict, file)
        # append results to file
        with open(f'{args.outfile}.json') as json_file:
            data = json.load(json_file)
            data.update(areas)

        with open(f'{args.outfile}.json', 'w') as outfile1:
            json.dump(data, outfile1)
        
        # create debug images for visual check
        pcv.params.debug = "print"
        pcv.params.debug_outdir = "./redos/0527"

        if area_dict['scale'] != 0:
            objs = roi_objects + [id_scale]
            scale_hier = np.array([[hier]])
            hs = np.append(hierarchy, scale_hier, axis=1)
            show_objects.show_objects(img, objs, hs, label)
            area_dict['flag'] = "PASS"
        else:
            # in this case, a circular scale was never found
            # debug image is in red to see what the heck was IDed instead
            objs = roi_objects + roi_scale
            try:
                hs = np.append(hierarchy, scale_hierarchy, axis=1)
            except ValueError:
                hs = hierarchy
                objs = roi_objects
            label = label+"_FAIL"
            show_objects.show_objects(img, objs, hs, label, red = 255)
            print(f'warning: scale may not be detected properly for {args.image}')
            area_dict['flag'] = "SCALE"
    else: print(f"FAILED AT {photo}")

if __name__ == "__main__":
    main()

