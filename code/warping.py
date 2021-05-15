# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:36:07 2020

@author: Elias
"""
import scipy.spatial.distance
import numpy as np
import math
import cv2

# getting aspectRatio based on following paper (page 31 and some pages before that):
# https://www.microsoft.com/en-us/research/publication/whiteboard-scanning-image-enhancement/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fzhang%2Fpapers%2Ftr03-39.pdf
# https://stackoverflow.com/questions/38285229/calculating-aspect-ratio-of-perspective-transform-destination-image

# Give points sorted like this: [[0,H],[W,H],[W,0],[0,0]]
def warpToRectangle(img, p, height, width):
    # image center
    #h, w = img.shape[:2]
    h, w = height, width
    u0 = (h) / 2.0
    v0 = (w) / 2.0
    
    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])
    
    contour = h1+h2+w1+w2
    if contour >= 150:
    
        w = max(w1, w2)
        h = max(h1, h2)
    
        # visible aspect ratio
        ar_vis = float(w) / float(h)
    
        # make numpy arrays and append 1 for linear algebra
        #    m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
        #    m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
        #    m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
        #    m4 = np.array((p[3][0],p[3][1],1)).astype('float32')
        m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
        m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
        m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
        m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')
        
        # Points are differently sorted than input points: Fix this:
        temp = m3
        m3 = m4
        m4 = temp
    
        # calculate the focal distance
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    
        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1
    
        n21 = n2[0]
        n22 = n2[1]
        n23 = n2[2]
    
        n31 = n3[0]
        n32 = n3[1]
        n33 = n3[2]
    
        #f = 1500
        f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))
        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')
    
        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)
    
        # calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))
        #    div = 5
        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)
    
        pts1 = np.array(p).astype('float32')  # cornerpoints of unwarped image
        pts2 = np.float32([[0, H], [W, H], [W, 0], [0, 0]])  # cornerpoints of new warped image
        #original pattern: [0,0],[W,0],[0,H],[W,H]]
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (W, H))
    
#        print("focal length:",f)
#        cv2.imshow('img',m.drawPolygon(img, p))
#        cv2.imshow('dst',dst)
#        cv2.waitKey(0)
        return dst
    else:
        raise ValueError
