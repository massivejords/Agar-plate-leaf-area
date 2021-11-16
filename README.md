# Measuring leaf area from agar plate images
This repository contains python scripts used to measure leaf area of A. thaliana plants grown inside agar plates and python notebooks demonstrating the workflow used to quantify leaf area.
## Demos
There are two demos included as ipython notebooks. The first is "Workflow_demo_sticker_and_plate.ipynb" This demonstrates the workflow used to crop the image to the plate, identify the leaves and count pixels, and to identify the area calibration scale. It also demonstrates how the agar plate itself can be used as a scale but this method was not tested on the whole image set. The validation_imgs.ipynb demonstrates how the test images using paper leaves were measured for validation through comparison with Easyleafarea (https://github.com/heaslon/Easy-Leaf-Area)
## Leaf area measurement scripts
The file "single_image_single_output" is a demo script that takes a single image as an argument and prints the leaf areas of all seedlings. To run it the image file must be specified as an argument. For example to run this on a test image from the included test images folder you could run the followling command (PowerShell):
```PowerShell
python3 .\single_image_single_output.py -i .\test_images\IMG_7454.JPG
```
This will print out the scale area in pixels and the dictionary containing the leaf area and position of all 6 seedlings. It also contains additonal information including if the measurement is suspicious. Suspicious is defined as a leaf area measurement taking up too little of the minimum enclosing circle. This indicates that glare or roots may have been picked up instead of leaves. The dictionary output of this demo could then be used to export the leaf area data to an appropriate format like JSON or CSV. This is shown in leaf_area.py. leaf_area.py is the batch image analysis script we used to measure leaf area in a large image set of arabidopsis seedlings. It is included for demonstration and will not run because the required data files are not available.

## Dependencies and included functions
The notebooks and scripts described above rely on the following packages: PlantCV, OpenCV (cv2), NumPy, and Matplotlib. They are designed to be run using Python 3. Additionally they rely on fuctions included in this repository in the files cluster_jordan.py, object_composition.py and show_objects.py.