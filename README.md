# Augmented Reality(AR) | Drawing images & 3D cubes on fiducials
[![License MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/RajPShinde/Augmented-Reality-AR-Drawing_images_and_3D_cubes_on_fiducials/blob/master/LICENSE)

## Authors
* **Raj Prakash Shinde** [GitHub](https://github.com/RajPShinde)
* **Shubham Sonawane** [GitHub](https://github.com/shubham1925)
* **Prasheel Renkuntla** [GitHub](https://github.com/Prasheel24)

## Output
<p align="center">
<h5>Output</h5>
<img src="/Output/AR.png">
</p>

<p align="center">
<h5>Tag ID</h5>
<img src="/Output/tagID.gif">
</p>

## Dependencies
* Ubuntu 16
* Python 3.7
* OpenCV 4.2
* Numpy
* copy
* sys
* argparse

## Run

To run the detection, lena super impose and to draw a cube on a video with single tag -

```
python3.7 singleTag.py --vid Tag0.mp4 --func 1
```
For video with multiple tags, run the below command -
```
python3.7 multiTag.py --vid multipleTags.mp4 --func 1
```

Options for "vid" argument -
* path to the video file

Options for "func" argument -
* 1 - to detect tag id and orientation
* 2 - to superimpose lena
* 3 - to draw a virtual cube on top of the tag

Input 1 or 0 in the command window (when prompted) to save into a video file

## Reference
* https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
* https://www.owlnet.rice.edu/~elec539/Projects97/morphjrks/warpsri.html