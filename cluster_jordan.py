import os
import cv2
import numpy as np
from plantcv.plantcv import print_image
from plantcv.plantcv import plot_image
from plantcv.plantcv import color_palette
from plantcv.plantcv import params
import sys
from datetime import datetime
from plantcv.plantcv import apply_mask



def cluster_contours(img, roi_objects, roi_obj_hierarchy, nrow=1, ncol=1, show_grid=False):

    """
    This function take a image with multiple contours and clusters them based on user input of rows and columns

    Inputs:
    img                     = RGB or grayscale image data for plotting
    roi_objects             = object contours in an image that are needed to be clustered.
    roi_obj_hierarchy       = object hierarchy
    nrow                    = number of rows to cluster (this should be the approximate  number of desired rows
                              in the entire image (even if there isn't a literal row of plants)
    ncol                    = number of columns to cluster (this should be the approximate number of desired columns
                              in the entire image (even if there isn't a literal row of plants)
    show_grid               = if True then the grid will get plot to show how plants are being clustered

    Returns:
    grouped_contour_indexes = contours grouped
    contours                = All inputed contours

    :param img: numpy.ndarray
    :param roi_objects: list
    :param roi_obj_hierarchy: numpy.ndarray
    :param nrow: int
    :param ncol: int
    :param show_grid: bool
    :return grouped_contour_indexes: list
    :return contours: list
    :return roi_obj_hierarchy: list
    """

    params.device += 1

    if len(np.shape(img)) == 3:
        iy, ix, iz = np.shape(img)
    else:
        iy, ix, = np.shape(img)

    # get the break groups
    if nrow == 1:
        rbreaks = [0, iy]
    else:
        rstep = np.rint(iy / nrow)
        rstep1 = np.int(rstep)
        rbreaks = range(0, iy, rstep1)
    if ncol == 1:
        cbreaks = [0, ix]
    else:
        cstep = np.rint(ix / ncol)
        cstep1 = np.int(cstep)
        cbreaks = range(0, ix, cstep1)

    # categorize what bin the center of mass of each contour
    def digitize(a, step):
        # The way cbreaks and rbreaks are calculated, step will never be an integer
        # if isinstance(step, int):
        #     i = step
        # else:
        num_bins = len(step)
        for x in range(0, num_bins):
            if x == 0:
                if a >= 0 and a < step[1]:
                    return 1
            else:
                if a >= step[x - 1] and a < step[x]:
                    return x
                elif a >= np.max(step):
                    return num_bins

    dtype = [('cx', int), ('cy', int), ('rowbin', int), ('colbin', int), ('index', int)]
    coord = []
    for i in range(0, len(roi_objects)):
        m = cv2.moments(roi_objects[i])
        if m['m00'] == 0:
            pass
        else:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            colbin = digitize(cx, cbreaks)
            rowbin = digitize(cy, rbreaks)
            a = (cx, cy, colbin, rowbin, i)
            coord.append(a)
    coord1 = np.array(coord, dtype=dtype)
    coord2 = np.sort(coord1, order=('colbin', 'rowbin'))

    # get the list of unique coordinates and group the contours with the same bin coordinates
    groups = []
    for i, y in enumerate(coord2):
        col = y[3]
        row = y[2]
        location = str(row) + ',' + str(col)
        groups.append(location)
    
    
    #unigroup = np.unique(groups)
    unigroup = ["1,1", "2,1", "3,1", "4,1", "5,1", "6,1"]
    coordgroups = []

    for i, y in enumerate(unigroup):
        col_row = y.split(',')
        col = int(col_row[0])
        row = int(col_row[1])
        for a, b in enumerate(coord2):
            if b[2] == col and b[3] == row:
                grp = i
                contour = b[4]
                coordgroups.append((grp, contour))
            else:
                pass
        if len(coordgroups) == 0:
            coordgroups.append((i, None))
        else:
            check = coordgroups[-1]
            if i != check[0]:
                coordgroups.append((i, None))
    
    coordlist = [[y[1] for y in coordgroups if y[0] == x] for x in range(0, (len(unigroup)))]
    
    contours = roi_objects
    grouped_contour_indexes = coordlist

    # Debug image is rainbow printed contours

    if params.debug is not None:
        if len(np.shape(img)) == 3:
            img_copy = np.copy(img)
        else:
            iy, ix = np.shape(img)
            img_copy = np.zeros((iy, ix, 3), dtype=np.uint8)

        rand_color = color_palette(len(coordlist))
        for i, x in enumerate(coordlist):
            for a in x:
                if a != None:
                    if roi_obj_hierarchy[0][a][3] > -1:
                        pass
                    else:
                        cv2.drawContours(img_copy, roi_objects, a, rand_color[i], -1, hierarchy=roi_obj_hierarchy)
        if show_grid:
            for y in rbreaks:
                cv2.line(img_copy, (0, y), (ix, y), (255, 0, 0), params.line_thickness)
            for x in cbreaks:
                cv2.line(img_copy, (x, 0), (x, iy), (255, 0, 0), params.line_thickness)
        if params.debug == 'print':
            print_image(img_copy, os.path.join(params.debug_outdir, str(params.device) + '_clusters.png'))
        elif params.debug == 'plot':
            plot_image(img_copy)

    return grouped_contour_indexes, contours, roi_obj_hierarchy

def cluster_contour_splitimg(rgb_img, grouped_contour_indexes, contours, hierarchy, outdir=None, file=None,
                             filenames=None):

    """
    This function takes clustered contours and splits them into multiple images, also does a check to make sure that
    the number of inputted filenames matches the number of clustered contours.

    Inputs:
    rgb_img                 = RGB image data
    grouped_contour_indexes = output of cluster_contours, indexes of clusters of contours
    contours                = contours to cluster, output of cluster_contours
    hierarchy               = hierarchy of contours, output of find_objects
    outdir                  = out directory for output images
    file                    = the name of the input image to use as a plantcv name,
                              output of filename from read_image function
    filenames               = input txt file with list of filenames in order from top to bottom left to right
                              (likely list of genotypes)

    Returns:
    output_path             = array of paths to output images

    :param rgb_img: numpy.ndarray
    :param grouped_contour_indexes: list
    :param contours: list
    :param hierarchy: numpy.ndarray
    :param outdir: str
    :param file: str
    :param filenames: str
    :return output_path: str
    """

    params.device += 1

    sys.stderr.write(
        'This function has been updated to include object hierarchy so object holes can be included\n')

    i = datetime.now()
    timenow = i.strftime('%m-%d-%Y_%H:%M:%S')

    if file is None:
        filebase = timenow
    else:
        filebase = os.path.splitext(file)[0]

    if filenames is None:
        namelist = []
        for x in range(0, len(grouped_contour_indexes)):
            namelist.append(x)
    else:
        with open(filenames, 'r') as n:
            namelist = n.read().splitlines()
        n.close()

    # make sure the number of objects matches the namelist, and if not, remove the smallest grouped countor
    # removing contours is not ideal but the lists don't match there is a warning to check output
    if len(namelist) == len(grouped_contour_indexes):
        corrected_contour_indexes = grouped_contour_indexes
    elif len(namelist) < len(grouped_contour_indexes):
        print("Warning number of names is less than number of grouped contours, attempting to fix, to double check "
              "output")
        diff = len(grouped_contour_indexes) - len(namelist)
        size = []
        for i, x in enumerate(grouped_contour_indexes):
            totallen = []
            for a in x:
                g = i
                la = len(contours[a])
                totallen.append(la)
            sumlen = np.sum(totallen)
            size.append((sumlen, g, i))

        dtype = [('len', int), ('group', list), ('index', int)]
        lencontour = np.array(size, dtype=dtype)
        lencontour = np.sort(lencontour, order='len')

        rm_contour = lencontour[diff:]
        rm_contour = np.sort(rm_contour, order='group')
        corrected_contour_indexes = []

        for x in rm_contour:
            index = x[2]
            corrected_contour_indexes.append(grouped_contour_indexes[index])

    elif len(namelist) > len(grouped_contour_indexes):
        print("Warning number of names is more than number of  grouped contours, double check output")
        diff = len(namelist) - len(grouped_contour_indexes)
        namelist = namelist[0:-diff]
        corrected_contour_indexes = grouped_contour_indexes

    # create filenames
    group_names = []
    group_names1 = []
    for i, x in enumerate(namelist):
        plantname = str(filebase) + '_' + str(x) + '_p' + str(i) + '.png'
        maskname = str(filebase) + '_' + str(x) + '_p' + str(i) + '_mask.png'
        group_names.append(plantname)
        group_names1.append(maskname)

    # split image
    output_path = []
    output_imgs = []
    output_masks = []

    for y, x in enumerate(corrected_contour_indexes):
        if outdir is not None:
            savename = os.path.join(str(outdir), group_names[y])
            savename1 = os.path.join(str(outdir), group_names1[y])
        else:
            savename = os.path.join(".", group_names[y])
            savename1 = os.path.join(".", group_names1[y])
        iy, ix, iz = np.shape(rgb_img)
        mask = np.zeros((iy, ix, 3), dtype=np.uint8)
        masked_img = np.copy(rgb_img)
        for a in x:
            if a != None:
                if hierarchy[0][a][3] > -1:
                    cv2.drawContours(mask, contours, a, (0, 0, 0), -1, lineType=8, hierarchy=hierarchy)
                else:
                    cv2.drawContours(mask, contours, a, (255, 255, 255), -1, lineType=8, hierarchy=hierarchy)

        mask_binary = mask[:, :, 0]

        if np.sum(mask_binary) == 0:
            pass
        else:
            retval, mask_binary = cv2.threshold(mask_binary, 254, 255, cv2.THRESH_BINARY)
            masked1 = apply_mask(masked_img, mask_binary, 'white')
            output_imgs.append(masked1)
            output_masks.append(mask_binary)
            if outdir is not None:
                print_image(masked1, savename)
                print_image(mask_binary, savename1)
            output_path.append(savename)

            if params.debug == 'print':
                print_image(masked1, os.path.join(params.debug_outdir, str(params.device) + '_clusters.png'))
                print_image(mask_binary, os.path.join(params.debug_outdir, str(params.device) + '_clusters_mask.png'))
            elif params.debug == 'plot':
                plot_image(masked1)
                plot_image(mask_binary, cmap='gray')

    return output_path, output_imgs, output_masks
