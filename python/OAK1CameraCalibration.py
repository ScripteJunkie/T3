#!/usr/bin/env python

import cv2
import numpy as np
import os

# USER INPUT VARIABLES

# Show or hide debug statements
from cv2 import findCirclesGrid

verbose = True
# Show or hide preview images
imgVerbose = True
# Defining the dimensions of checkerboard
CHECKERBOARD = (45, 31)
# Directory where images are located (Directory must be located in same directory as code)
imgDirName = 'images'

#######################################################################################################################

curDir = os.path.dirname(__file__)
imgDir = os.path.join(curDir, imgDirName)

if verbose:
    print(
        f'Current Directory: {curDir}\n'
        f'Image Directory:   {imgDir}'
    )

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
# Creating vector to store file names for each checkerboard image
imgList = []
# Set the image directory

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# List files in img directory
fileList = os.listdir(imgDir)

# Extracting path of images stored in a given directory
for file in fileList:
    # Make sure file is an image
    if file.endswith('.jpeg'):
        img_path = os.path.join(imgDir, file)
        # Adding each image to the image list
        imgList.append(img_path)
        if verbose:
            print(img_path)

for imagePath in imgList:
    img = cv2.imread(imagePath)
    imgW = img.shape[1]
    imgH = img.shape[0]

    if imgVerbose:
        
        
        scaledDim = (int(round(imgW / 2)), int(round(imgH / 2)))
        previewImg = cv2.resize(img, scaledDim)
        cv2.imshow('img', previewImg)
        cv2.waitKey(0)
        cv2.destroyWindow('img')

    # (hMin = 0 , sMin = 0, vMin = 160), (hMax = 179 , sMax = 20, vMax = 255)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, (0, 0, 160), (179, 20, 255))
    # result = cv2.bitwise_and(img, img, mask=mask)
    # bgrResult = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    grayResult = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if imgVerbose:
        scaledDim = (int(round(imgW / 2)), int(round(imgH / 2)))
        previewGrayImg = cv2.resize(grayResult, scaledDim)
        cv2.imshow('grayResult', previewGrayImg)
        cv2.waitKey(0)
        cv2.destroyWindow('grayResult')

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, centers = findCirclesGrid(grayResult, (46, 32), )
    # ret, corners = cv2.findChessboardCorners(grayResult, CHECKERBOARD,
    #                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    print(centers)
    # print(type(corners))

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        print("Circles found!")
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(grayResult, centers, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        if imgVerbose:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    if imgVerbose:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""

size = (img.shape[1], img.shape[0])
print(size)

# Obtaining camera calibration matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

if verbose:
    print(
        f'Camera matrix:\n'
        f'{mtx}\n'
        f'dist:\n'
        f'{dist}\n'
        f'rvecs:\n'
        f'{rvecs}\n'
        f'tvecs:\n'
        f'{tvecs}'
    )

for imagePath in imgList:
    img = cv2.imread(imagePath)
    cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), thickness=4)
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
    # crop the image
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    dst = dst[y:y+h, x:x+w]

    scaledDim = (int(round(img.shape[1] / 2)), int(round(img.shape[0] / 2)))

    combinedImg = cv2.addWeighted(img, 0.4, dst, 0.1, 0)
    # combinedScaled = cv2.resize(combinedImg, scaledDim)

    # scaledImg = cv2.resize(img, scaledDim)
    # scaledDst = cv2.resize(dst, scaledDim)

    if imgVerbose:
        cv2.imshow('Combined Image', combinedImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

testImg = cv2.imread(r'C:\Users\raman\PycharmProjects\T3\python\Screenshot 2022-03-24 15-36-46.png')
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
fixedTestImg = cv2.undistort(testImg, mtx, dist, None, newCameraMtx)

cv2.imshow('Fixed Image', fixedTestImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('savedTestImage.png', fixedTestImg)

