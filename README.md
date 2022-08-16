# Measuring leaf area from agar plate images
This repository contains python scripts used to measure leaf area of A. thaliana plants grown inside agar plates and python notebooks demonstrating the workflow used to quantify leaf area.

## Dependencies and included functions
The notebooks and scripts described above rely on the following packages: PlantCV, OpenCV (cv2), NumPy, and Matplotlib. They are designed to be run using Python 3. 
to quickly install dependencies:
```shell
pip install -r requirements.txt
```
Additionally they rely on fuctions included in this repository in the files cluster_jordan.py, object_composition.py and show_objects.py.
For more information about jupyer notebooks please see https://jupyter.org/

## Demos
There are two demos included as jupyter notebooks. The first is ```dedicated_scale_demo.ipynb``` This demonstrates the workflow used to crop the image to the plate, identify the leaves and count pixels, and to identify the area calibration scale. The second is ```plate_as_scale_demo.ipynb``` In addition to doing all the same things as the previous It also demonstrates how the agar plate itself can be used as a scale. The test images for each demo are found in the test-images folder. Each demo has a seperate set of test images since these workflows were developed for different image sets.

## Additional files
As part of our validation of this approach we compared rosette area measured from plate images to rosette area measured from excised seedlings that were placed onto paper and photographed from directly overhead. The workflow for measuring rosette area of the excised seedlings can be found in "rosettes_on_paper.ipynb"
```Batch_analysis.py``` is the batch image analysis script we used to measure leaf area in a large image set of arabidopsis seedlings. It is included for reference purposes only and will not run because the required data files are not available.

