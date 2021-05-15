# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:49:04 2020

@author: Elias
"""
import numpy as np
import cv2

def savePickleDescriptors(keypoints, descriptors):   # function to workaround saving keypoints in a pickle
    i = 0
    temp_lijst = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i = i + 1
        temp_lijst.append(temp)
    return temp_lijst

def getPickleDescriptors(lijst):
    keypoints, descriptors = [],[]
    for point in lijst:
        feature_temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2],
                                    _response=point[3], _octave=point[4], _class_id=point[5])
        descriptor_temp = point[6]
        keypoints.append(feature_temp)
        descriptors.append(descriptor_temp)
    return keypoints, np.array(descriptors)