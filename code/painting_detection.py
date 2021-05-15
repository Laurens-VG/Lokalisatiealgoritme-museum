# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:36:51 2020

@author: Elias
"""

import cv2
import numpy as np
import math
import config as cf
import warping as w
import sys
import os
import timeit

# for sorting points:
origin = [0, 1000]  # is adaptive to the detected polygon cornerpoints
refvec = [-1, 0]  # previous values: origin: 0,1000 and refvec = 0,1

def showImage(img, res=[], name='Source', name2='Result', delay=0, std_div = 1):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    h, w = img.shape[:2]
    if w > 1920 or h > 1080:
        divider = 5
    else:
        divider = std_div
    h, w = int(h / divider), int(w / divider)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, img)
    if len(res) > 0:
        cv2.namedWindow(name2, cv2.WINDOW_NORMAL)
        h2, w2 = res.shape[:2]
        h2, w2 = int(h2 / divider), int(w2 / divider)
        cv2.resizeWindow(name2, w2, h2)
        cv2.imshow(name2, res)
    key = cv2.waitKey(delay)
    if delay == 0:
        cv2.destroyAllWindows()
    if key == 121:  #y-key
        return 'y'
    elif key == 110: #n-key
        return 'n'
    return ''

def readImage(path_img):
    img = cv2.imread(path_img, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

def sharpenImage(img):
    #sharpness = variance_of_laplacian(img)
    #print(sharpness)
    #if sharpness < 25:
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def getContours(img_gray, blur_it=51):
    """
    LINK: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    """
    #blur_it = 15
    blurred = cv2.GaussianBlur(img_gray, (351, 351), blur_it)  # (501,501),33)
    # thresh = cv2.Canny(img_gray,600,600,apertureSize = 5)
    middle_val = np.mean(img_gray)
    if middle_val <= 100:
        method = cv2.THRESH_BINARY
    else:
        method = cv2.THRESH_BINARY_INV
    ret, thresh = cv2.threshold(blurred, middle_val, 255, method)  # 170 #cv2.THRESH_TRIANGLE
    # thresh = removeInnerWhite(thresh)
    cntrs, hrchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cf.show_extraction:
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        img_gray = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(thresh, cntrs, -1, (0,255,0), 5)
        cv2.namedWindow("Thresholding", cv2.WINDOW_NORMAL)
        h2, w2 = thresh.shape[:2]
        h2, w2 = 450, 1500
        cv2.resizeWindow("Thresholding", w2, h2)
        cv2.imshow("Thresholding", np.hstack((img_gray, thresh)))
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
    return cntrs, hrchy

def getContoursAdaptive(img, img_gray, blur_it=51):
    #Same as getContours but with adaptive thresholding
    blurred = cv2.GaussianBlur(img_gray, (501, 501), blur_it)  # (501,501),33)
    # print("img.dtype:", img.dtype)
    # thresh = cv2.Canny(img,600,600,apertureSize = 5)
    middle_val = np.mean(img_gray)
    if middle_val <= 100:
        method = cv2.THRESH_BINARY
    else:
        method = cv2.THRESH_BINARY_INV
    thresh = cv2.adaptiveThreshold(blurred,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,801,3)
    cv2.namedWindow("blur",cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("blur", 1200,800)
    cv2.imshow("blur", thresh)
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1200,800)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # thresh = removeInnerWhite(thresh)
    cntrs, hrchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cntrs, hrchy


def getAllPolygons(cntrs, height, width):  # gets polygon of largest contour (biggest surface)
    cornerpts = []
    if len(cntrs) != 0:  
        #draw in blue the contours that were found
#        cv2.drawContours(res_all, cntrs, -1, 255, 5)
        # find the biggest countour (c) by the area
        sorted_cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)
        # print("sorted_cntrs", sorted_cntrs[0].reshape(-1,2))
        for contour in sorted_cntrs:
            c = contour.reshape(-1, 2)
            # x,y,w,h = cv2.boundingRect(c) #TODO: change rectangle to trapezium
            if not( isFloorOrCeiling(c, height, width)):
                epsilon = 0.1 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                cornerpts.append(approx)
    return cornerpts

def getLargestPolygon(img, cntrs):  # gets polygon of largest contour (biggest surface)
    height, width = img.shape[:2]
    if len(cntrs) != 0:
        # find the biggest countour (c) by the area
        sorted_cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)
        # print("sorted_cntrs", sorted_cntrs[0].reshape(-1,2))
        c = sorted_cntrs[0].reshape(-1, 2)
        # x,y,w,h = cv2.boundingRect(c) #TODO: change rectangle to trapezium
        if isFloor(c, height, width) or isCeiling(c, height, width):
            c = sorted_cntrs[1]
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        #        print("h", height, "w", width)
        #        print("corner_pts", approx)
    #        # draw the biggest contour (c) in green
    #        cv2.rectangle(res_largest,(x,y),(x+w,y+h),(0,255,0),5)
    # showImage(res_all, res_approx, name="getLargestPolygon")
    return approx


def isFloorOrCeiling(c_pts, height, width):
    i, cnt_f, cnt_c = 0, 0, 0
    while cnt_f < 2 and cnt_c < 2 and i < len(c_pts):
        pt_h = c_pts[i][1]
        if abs(pt_h - height) <= 3:
            cnt_f += 1
        elif pt_h <= 3:
            cnt_c += 1
        i += 1
    if cnt_f >= 2 or cnt_c >= 2:
        #print("Floor detected, using next polygon...")
        return True
    else:
        return False

def isFloor(c_pts, height, width):
    i, cnt = 0, 0
    while cnt < 2 and i < len(c_pts):
        _, pt_h = c_pts[i][0], c_pts[i][1]
        if abs(pt_h - height) <= 3:
            cnt += 1
        i += 1
    if cnt >= 2:
        #print("Floor detected, using next polygon...")
        return True
    else:
        return False


def isCeiling(c_pts, height, width):
    i, cnt = 0, 0
    while cnt < 2 and i < len(c_pts):
        _, pt_h = c_pts[i][0], c_pts[i][1]
        if pt_h <= 3:
            cnt += 1
        i += 1
    if cnt >= 2:
        #print("Ceiling detected, using next polygon...")
        return True
    else:
        return False


def sortCorners(pts):  # sorts points based on 'origin' and 'refvec' parameters
    # pts = [[2,3], [5,2],[4,1],[3.5,1],[1,2],[2,1],[3,1],[3,3],[4,3]]
    pts = np.reshape(pts, (-1, 2)).tolist()
    pts = sorted(pts, key=clockwiseangle_and_distance)
    # print("pts sorted:", pts)
    return np.array(pts, dtype=np.float32)


def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Neg. angles represent counter-clockwise angles so subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def getMiddlePoint(poly_corners):
    poly_corners = np.reshape(poly_corners, (-1, 2)).tolist()
    w_list, h_list = map(list, zip(*poly_corners))
    return [(max(w_list) + min(w_list)) / 2, (max(h_list) + min(h_list)) / 2]


def variance_of_laplacian(img): #Returns a sharpness value, higher is sharper.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value


def getSharpFrames(video_path, framerange, step=30):
    start_time = timeit.default_timer()
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = framerange[0], framerange[1]
    if end == ':': end = total_frames-1
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in video:",int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    if start < end and end <= total_frames: #video.get(7):  # .get(7) = total number of frames in video
        frames, frame_nrs, values = [], [], []
        max_frames = end - start
        cnt = 0
        print("Checking", max_frames,"frames with stepsize", step, "...")
        video.set(1, start)
        nr = start
        for i in range(start, end, step):
            video.grab()
            while nr != i:
                video.grab()
                nr += 1
            status, frame = video.retrieve()
#            video.set(1, i)
#            ret, frame = video.read()
            progress = (cnt+1)*step*100/max_frames
            print("\r[%-25s] %d%%" % ('='*int(progress/4), round(progress-1,0)),end="", flush=True)
            value = variance_of_laplacian(frame)
#            print(value)
            if value >= cf.sharpness_threshold:  # see config file
                # frames.append(frame) #holding all frames in a variable creates memory issues!
                frame_nrs.append(i)
                values.append(value)
            cnt += 1
            nr += 1
        if len(frame_nrs) < 80:
            print("\nNot enough sharp frames found. Returning every "+str(step)+"th frame")
            frame_nrs = np.arange(0, total_frames, step)
        else:
            print("\n" +str(len(frame_nrs)), "sharp frames found. Stay Safe xx")
        if cf.show_runtime_modules:
            print('getSharpFrames: ', timeit.default_timer() - start_time)
        video.release()
        
#        for k in range(50,240,15):
#            frame_nrs.append(total_frames-k)
        
        return frames, frame_nrs, values
    else:
        print("ERROR: wrong videopath or framerange given. Aborting...")
        video.release()
        sys.exit(0)

def getSharpFrame(video_path, frame_nr):
    #start_tijd = timeit.default_timer()
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_nr)  # Added by elias
    ret, frame = cap.read()
    sharpness = variance_of_laplacian(frame)
    return frame, sharpness

def calculateBlurIterations(height, width):
    blur_it = int(height*width/cf.blur_it_factor)
    print("blur_it", blur_it)
    return blur_it
        
def getVideoProperties(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(video.get(cv2.CAP_PROP_FPS),0)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video.release()
    return total_frames, fps, h, w

def getImageFromVideo(video_path, frame_nr):
    video = cv2.VideoCapture(video_path)
    video.set(1, frame_nr)
    _, frame = video.read()
    video.release()
    return frame

def removeInnerWhite(img):  # Seems to be a problem where the whole image is shifted down after this method
    res = cv2.erode(img, (3, 3), iterations=350)
    res = cv2.dilate(res, (3, 3), iterations=350)
    # showImage(img, res)
    return res

def improveDarkImages(img):
    h, w, c = img.shape
    brightness = int(np.sum(img)/(c*w*h))
    #print("brightness", brightness)
    if brightness <= 60:
        add = 70-brightness
        img = img[:,:,:]+add
    return img

# def warpToRectangle(img, poly_corners, relation="4:3", width=2016):
#    poly_corners = np.array(poly_corners, dtype=np.float32)
#    tar_dim = relation.strip().split(':')
#    tar_w, tar_h = int(tar_dim[0]), int(tar_dim[1])
#    #TODO: sort corners from top right to bottom left
#    print("poly_corners: ", poly_corners)
#    h, w = img.shape[:2]
#    poly_corners = sortCorners(poly_corners)
#    crds_dst = np.array([[70,h-8.],[w-70,h-8.],[w-70.,70.],[70.,70.]], dtype=np.float32)
#    trf_matr = cv2.getPerspectiveTransform(poly_corners, crds_dst)
#    res = cv2.warpPerspective(img, trf_matr, (w, h))
#    return res

# def getContourOld(path_img):
#    img = cv2.imread(cf.path_database + path_img,1)
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    res = cv2.imread(cf.path_database + path_img,1)
#    blurred = cv2.GaussianBlur(img_gray,(501,501),33)
#    showImage(blurred)
#    ret,thresh = cv2.threshold(blurred,170,255,0)
#    showImage(thresh)
#    corners = cv2.goodFeaturesToTrack(thresh,20,0.01,20)
#    corners = np.int0(corners)
#    for i in corners:
#        x,y = i.ravel()
#        cv2.circle(res,(x,y),25,(0,0,255),-1)
#    showImage(res)
#    return 0