# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:32:38 2020

@author: Elias, Reiner
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import config as cf
import painting_detection as pd
import warping as w
import pickle_tools as tls
import sys, os
import timeit
from localization import show_individual_room

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity


def histogramMatching(imageA, imageB):
    hsvA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

    histA = cv2.calcHist([hsvA], [0, 1], None, [180, 255], [0, 180, 0, 255])
    histA = cv2.normalize(histA, histA, 0, 1, cv2.NORM_MINMAX, -1)
    histB = cv2.calcHist([hsvB], [0, 1], None, [180, 255], [0, 180, 0, 255])
    histB = cv2.normalize(histB, histB, 0, 1, cv2.NORM_MINMAX, -1)

    correl = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    chisqr = cv2.compareHist(histA, histB, cv2.HISTCMP_CHISQR)
    inters = cv2.compareHist(histA, histB, cv2.HISTCMP_INTERSECT)
    bhat = cv2.compareHist(histA, histB, cv2.HISTCMP_BHATTACHARYYA)

    return correl, chisqr, inters, bhat


def similarityImages(img_to_match, db_img):
    # ONLY USEFULL IF IMAGES HAVE SAME SHAPE...
    db_h, db_w = db_img.shape[:2]
    h, w = img_to_match.shape[:2]
    scaling_factors = np.array([db_h/h, db_w/w])
    img_to_match = cv2.resize(img_to_match, None, fx=scaling_factors[1], fy=scaling_factors[0], 
                              interpolation=cv2.INTER_AREA)
    # compute the mean squared error and structural similarity
    # index for the images
    m = mean_squared_error(db_img, img_to_match)
    s = structural_similarity(db_img, img_to_match, multichannel=True)
    
    # show the images
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    cv2.putText(db_img, "m: "+str(m), (30, 30), font, 1, (255, 0, 0))
#    cv2.putText(db_img, "s: "+str(s), (30, 60), font, 1, (255, 0, 0))
#    pd.showImage(img_to_match,db_img)
#    print("M: ", round(m,2),'\nS:', round(s,2))
    return m, s


def getFrame(video_path, frame_nr, showFrame=False):
    start_tijd = timeit.default_timer()
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_nr)  # Added by elias
    ret, frame = cap.read()
    if showFrame:
        cv2.imshow("Current frame", frame)
    cap.release()
    if cf.show_runtime_modules:
        print("getFrame:", timeit.default_timer()-start_tijd)
    return frame


def calcPerformance(success_frames, total_frames, correct, wrong):
    print("\n-------------- RESULTS ------------")
    if total_frames != 0:
        # Houdt nog geen rekening met of de match correct is of niet! (handmatige verificatie voor nodig)
        print("\nPrecision, painting extraction:")
        print("#sharp frames with match/#sharp frames =", str(success_frames) + '/' + str(total_frames),
              '=', round(success_frames / total_frames, 4))
    if correct + wrong != 0:
        print("Precision, matching:")
        print("#correct matches/#total matches =", str(correct) + '/' + str(correct + wrong), 
              '=', round(correct / (correct+wrong), 4))
    return 0


def downgradeDatabase(database):
    for i, path_db_img in enumerate(database):
        try:
            progress = (i + 1) * 100 / len(database)
            print("\rDowngrading database: [%-25s] %d%%" % ('=' * int(progress / 4),
                                                            round(progress, 0)), end="", flush=True)
            path_db_img = path_db_img.replace('\\', '/')
            db_img = cv2.imread(path_db_img)
            db_img = downgradeResolution(db_img, 5)
            directories = path_db_img.split('/')
            directories[-3] = "db_downgraded"
            new_path = ""
            for j in range(len(directories) - 1):
                new_path += directories[j] + '/'
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv2.imwrite(new_path + directories[-1], db_img)
        except cv2.error:
            print("Image", directories[-1:], "failed.")
            print("Are all given files images?")
    print("\nDone. Files saved in", directories[:-2])
    return 0


def downgradeResolution(img, factor=3):
    #TODO: Improve speed by downgrading based on original image size
    scaling_factor = float(1/factor)
    res = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(cf.MAX_FEATURES)
    kp = orb.detect(res_gray, None)
    if (len(kp) < 15 and factor > 1):
        #print("not enough keypoints:", len(kp), "with factor:", factor)
        res = downgradeResolution(img, factor-1)
    return res


def extractPaintings(frame, frame_nr, blur_it):
    start_tijd = timeit.default_timer()
    paintings, list_sorted_poly_corners = [], []
    fails = [[],[],[]]
    height, width = frame.shape[:2]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cntrs, _ = pd.getContoursAdaptive(frame, frame_gray, blur_it)
    cntrs, _ = pd.getContours(frame_gray, blur_it)
    # poly_corners = pd.getLargestPolygon(frame, cntrs)
    list_unsorted_poly_corners = pd.getAllPolygons(cntrs, height, width)
        
    for i, poly_corners in enumerate(list_unsorted_poly_corners, start=1):
        if len(poly_corners) == 4:
            pd.origin = pd.getMiddlePoint(poly_corners)
            sorted_poly_corners = pd.sortCorners(poly_corners)
            try: 
                #change frame to frame_gray in final build and delete conversion in databasematch
                res = w.warpToRectangle(frame, sorted_poly_corners, height, width)
                paintings.append(res)
                list_sorted_poly_corners.append(sorted_poly_corners)
            except IndexError:
                fails[0].append(i)
                #print("IndexError while warping. Image", i, "removed removed from frame.")
            except ValueError:
                fails[1].append(i)
                #print("Divide by zero in warping, line 64 (f)")
            except  RuntimeWarning:
                fails[1].append(i)
            except OverflowError:
                fails[1].append(i)
                #print("Divide by zero in warping, line 72 (ar_real)")
            except ZeroDivisionError:
                fails[1].append(i)
                #print("Divide by zero in warping, line 72 (ar_real)")
            except cv2.error:
                fails[2].append(i)
                #print("Opencv error in warping, line 90")
                # print("Message: {0}".format(err))
    #print errors
#    if len(fails[0]) > 0:
#        print("Polygons with IndexErrors:", fails[0][:])
#    if len(fails[1]) > 0:
#        print("Polygons with ZeroDivide in warping:", fails[1][:])
#    if len(fails[2]) > 0:
#        print("Polygons with OpencvError:", fails[2][:])
    if cf.show_runtime_modules:
        print("Extract paintings :", timeit.default_timer()-start_tijd)
    return paintings, list_sorted_poly_corners


def inkFrame(frame, frame_nr, poly_corners, sharpness=None, score=None):
    frame_inked = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_inked, "frame: " + str(frame_nr), (0, 30), font, 1, (0, 0, 255), thickness=2)
    if sharpness is not None:
        cv2.putText(frame_inked, "sharpness: " + str(round(sharpness, 4)), (0, 60), font, 1, (0, 0, 255), thickness=2)
    if score is not None:
        cv2.putText(frame_inked, "score: " + str(round(score, 4)), (0, 90), font, 1, (0, 0, 255), thickness=2)
    frame_inked = drawPolygon(frame_inked, poly_corners)
    return frame_inked


def drawPolygon(img, corners):
    # color = np.random.rand(3,)*255
    color = (0, 255, 0)
    cnt = len(corners)
    for i in range(cnt):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % cnt]), color, 4)
    return img


def findDatabaseMatch2(img, database, all_descriptors, print_scores=False):
    start_tijd = timeit.default_timer()
    # Same as findDatabaseMatch, but uses the local stored keypoints and descriptors
    # of database instead of calculating this every time again (which was very slow)
    scores = []
    db_no_matches = []
    kps_and_matches = []

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(cf.MAX_FEATURES)
    keypoints1 = orb.detect(img_gray, None)
    if len(keypoints1) < 10:
        img_gray = pd.sharpenImage(img_gray)
        keypoints1 = orb.detect(img_gray, None)
    keypoints1, descriptors1 = orb.compute(img_gray, keypoints1)
    #print("Dim painting:", img.shape, "| Detected keypoints:", len(keypoints1))

    for path_db_img in database:
        # Get ORB descriptors for this db_img.
        directories = path_db_img.split('/')
        painting_name = directories[-1].replace(".png", "")
        temp_list = all_descriptors.get(painting_name)
        if len(temp_list[0]) == 0:
            continue
        keypoints2, descriptors2 = temp_list[0], temp_list[1]
        try:
            # Match features.
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)

            # Only keep good matches (value less than threshold in config.py)
            new_matches = np.array([])
            for match in matches:
                if match.distance < cf.max_hamming_distance:
                    #new_matches.append(match)
                    new_matches = np.append(new_matches, match)
            matches = new_matches
            score = len(matches)
            # Remove line below in final build!
            #matches.sort(key=lambda x: x.distance, reverse=False)
            if score > cf.remove_matches_with_score:
                scores.append([path_db_img, score])
                if cf.show_matches:
                    kps_and_matches.append([keypoints1, keypoints2, matches])
            if print_scores:
                print("\npainting:", painting_name)
                print("score:", score)
                print("matches:", matches[:4])
        except cv2.error as err:
            print("cv2.error")
            print("msg: {0}".format(err))
        except ZeroDivisionError:
            db_no_matches.append(painting_name)
            # print("WARNING: No matches between keypoints found.")
    if len(db_no_matches) > 0:
        print("No matches found on:", db_no_matches)

    #print("Possible matches:", len(scores))
    if len(scores) == 0:
        #print("No match found.")
        return None, None, None
    else:
        bestMatch, bestIndex = getBestMatch(scores)
        #print("Match found with score:", bestMatch[1])
        match_results = None
        if cf.show_matches:
            match_results = kps_and_matches[bestIndex]
        if cf.show_runtime_modules:
            print("findDBmatch ("+str(len(keypoints1))+"):", timeit.default_timer()-start_tijd)
        return bestMatch[0], bestMatch[1], match_results


def getBestMatch(scores):
    # scores = [['k', 12],['w', 128],['s', 4], ['o', 127],  ['t', 501], ['p', 496]] #test
    sortedMatches = sorted(scores, key=lambda x: -x[1])
    bestMatch = sortedMatches[0]
    indexBestMatch = scores.index(bestMatch)
    if bestMatch[1] == 0:
        bestMatch = None
        indexBestMatch = None
    return bestMatch, indexBestMatch


def saveDescriptorFiles(database, overwrite_previous=0):
    if not os.path.exists(cf.path_descriptor_files) or overwrite_previous:
        bad_images = []
        orb = cv2.ORB_create(cf.MAX_FEATURES)
        for i, path_db_img in enumerate(database):
            progress = (i + 1) * 100 / len(database)
            print("\rStoring descriptors: [%-25s] %d%%: %s" % ('=' * int(progress / 4),
                                                               round(progress, 0), str(path_db_img[-22:])), end="",
                  flush=True)
            db_img = cv2.imread(path_db_img)
            # Convert db image to grayscale
            db_img_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
            # Detect ORB features and compute descriptor.
            keypoints2, descriptors2 = orb.detectAndCompute(db_img_gray, None)
            if len(keypoints2) != 0:
                pickle_file = tls.savePickleDescriptors(keypoints2, descriptors2)
                directories = path_db_img.split('/')
                directories[-3] += "_pickles"
                directories[-2] += "_pickles"
                directories[-1] = directories[-1].replace(".png", "_kp_descr.p")
                new_path = cf.path_descriptor_files + directories[-2] + '/'
                #        new_path = ""
                #        for j in range(len(directories)-1):
                #            new_path += directories[j] + '/'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path += directories[-1]
                pickle.dump(pickle_file, open(new_path, "wb"))
            else:
                bad_images.append(path_db_img[-22:])
        print('')
        if len(bad_images) > 0:
            print("No keypoints found for:", bad_images, '\n')
    else:
        print("Descriptor files already saved.")
    return 0


def loadDescriptorFiles(path_of_folder):
    print("Loading keypoints & descriptors...\n")
    all_descriptors = {}
    pickle_paths = makeListFromDatabase(path_of_folder)
    for path in pickle_paths:
        directories = path.split('/')
        pickle_file = pickle.load(open(path, "rb"))
        keypoints, descriptors = tls.getPickleDescriptors(pickle_file)
        painting_name = directories[-1].replace("_kp_descr.p", "")
        all_descriptors[painting_name] = [keypoints, descriptors]
    return all_descriptors


def showMatches(img2, img3, kps_and_matches):
    if kps_and_matches != None:
        kp1, kp2, matches = kps_and_matches
        #print("Matches:", [match.distance for match in matches])
        imMatches = cv2.drawMatches(img2, kp1, img3, kp2, matches, None)
        cv2.namedWindow("Found keypoint matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Found keypoint matches", 923, 615)
        cv2.imshow("Found keypoint matches", imMatches)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
    return 0

def toonCanvas(frame, img2, img3, img3_path, kps_and_matches, delay):
    result = ''
    copy_img2, copy_img3 = img2.copy(), img3.copy()

    if cf.show_frames:
        canvas = np.zeros((700, 1704, 3), np.uint8)
        h, w = frame.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        max_height = 700
        if max_height < h:
            scaling_factor = max_height / float(h)
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        canvas[0:h, 0:w] = frame
        max_width = ((1700 - w) / 2) - 4
        if max_height < h2 or max_width < w2:
            scaling_factor = max_height / float(h2)
            if max_width / float(w2) < scaling_factor:
                scaling_factor = max_width / float(w2)
            img2 = cv2.resize(img2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            h2, w2 = img2.shape[:2]
        canvas[0:h2, w + 2:(w + w2) + 2] = img2
        if max_height < h3 or max_width < w3:
            scaling_factor = max_height / float(h3)
            if max_width / float(w3) < scaling_factor:
                scaling_factor = max_width / float(w3)
            img3 = cv2.resize(img3, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            h3, w3 = img3.shape[:2]
        canvas[0:h3, w + w2 + 4:w + w2 + w3 + 4] = img3
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Room detected match: " + img3_path[img3_path.find("zaal_"):img3_path.find("/Zaal")], (1400, 600), font, 0.5, (255, 255, 255))
        # cv2.putText(canvas, "Press 'y' if correct match", (1400, 630), font, 0.5, (255, 255, 255))
        # cv2.putText(canvas, "Press 'n' if wrong match", (1400, 660), font, 0.5, (255, 255, 255))
        result = pd.showImage(canvas, name="Database Match", delay=1, std_div=1.4)
        
        if cf.show_matches:
            showMatches(copy_img2, copy_img3, kps_and_matches)
        
        show_individual_room(img3_path[img3_path.find("zaal_"):img3_path.find("/Zaal")])
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
    return result


def ORBmatching(im_fname, database, keypoint_count):  # Reiner
    """
        ORB matching test afbeelding = im_fname
        database = alle fnames in database
    """
    print("original image: " + im_fname)
    img1 = cv2.imread(im_fname)

    # cv2.imshow("ori", img1)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    keypoints1 = cv2.drawKeypoints(img1, kp1, None)
    # cv2.imshow("keypoints1", keypoints1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    goodMatchesMap = []
    goodImages = []
    for db_fname in database:
        try:
            img2 = cv2.imread(db_fname)
            print("testing image: " + db_fname)
            # cv2.imshow("test", img2)
            kp2, des2 = orb.detectAndCompute(img2, None)
            keypoints2 = cv2.drawKeypoints(img2, kp2, None)
            # cv2.imshow("keypoints2", keypoints2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # normal matches
            matches = bf.match(des1, des2)
            # matches = sorted(matches, key=lambda x: x.distance)
            # res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:keypoint_count], None, flags=2)

            res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            #            plt.imshow(res), plt.show()
            pd.showImage(res)
            """
            Distance attribute in DMatch is a measure of similarity between the two descriptors(feature vectors). 
            If the distance is less, then the images are more similar and vice versa.
            https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm
            """
            good = []
            lowe_ratio = 0.89
            for m in matches:
                if m.distance < lowe_ratio:
                    good.append(m)
            if len(good) > 0:
                goodMatchesMap.append([db_fname, len(good)])
                goodImages.append(res)

            print("using ORB with lowe_ratio: " + str(lowe_ratio))
            print("there are: " + str(len(good)) + " good matches")
            print("")

        except cv2.error as err:
            print("testing image: " + db_fname + " failed")
            print("no or not enough keypoints found")
            print("error message: {0}".format(err))
    return goodMatchesMap, goodImages


def makeListFromDatabase(path):
    #    database_subfiles = glob.glob("database_schilderijen/*")  #Voor Reiner
    database_subfiles = glob.glob(path)
    #    for i in range(len(database_subfiles)):
    #        database_subfiles[i] = database_subfiles[i].replace('\\', '/')
    database = np.empty(0)
    for subfile in database_subfiles:
        temp = glob.glob(subfile + "/*")
        # Fix '\\' which sometimes occurs when using glob
        for i in range(len(temp)):
            temp[i] = temp[i].replace('\\', '/')
        database = np.hstack((database, temp))
    return database

##OLD FUNCTIONS
# def findDatabaseMatch(img, database, hist_equalization=False, print_scores=False, show_matches=False):
#    scores = []
#    
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    print("Dim painting:", img.shape)
#    if hist_equalization:
#        img_eq = cv2.equalizeHist(img_gray)
#        img_gray = img_eq
#    orb = cv2.ORB_create(cf.MAX_FEATURES)
#    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
#    if descriptors1 is None:
#        print("No keypoints found on extracted painting.")
#        return None, None
#        #return np.zeros((50,50,3)), None
#    
#    for path_db_img in database:
#        db_img =  cv2.imread(path_db_img)
#        # Convert db image to grayscale
#        db_img_gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
#        if hist_equalization:
#            db_img_eq = cv2.equalizeHist(db_img_gray)
#            db_img_gray = db_img_eq
#            #pd.showImage(np.vstack((img_gray,img_eq)),np.vstack((db_img_gray,db_img_eq)),"Hist equalization")
#            
#        # Detect ORB features and compute descriptor.
#        keypoints2, descriptors2 = orb.detectAndCompute(db_img_gray, None)
#        try:
#            # Match features.
#            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#            matches = matcher.match(descriptors1, descriptors2, None)
#            # Sort matches by score
#            matches.sort(key=lambda x: x.distance, reverse=False)
#            # Remove not so good matches
#            numGoodMatches = int(len(matches) * cf.GOOD_MATCH_PERCENT)
#            matches = matches[:numGoodMatches]
#            #TODO: do something with numGoodMatches (bv >40 is goede match)
#            
#            if show_matches:
#                # Draw top matches
#                imMatches = cv2.drawMatches(img, keypoints1, db_img, keypoints2, matches, None)
#                pd.showImage(imMatches, name="matches", delay=500)
#            
#            # Calculate matching score (lower is better)
#            score = np.mean([match.distance for match in matches])
#            if score == score: #Check if score is not NaN
#                scores.append([path_db_img, score])
#            if print_scores:
#                print("\npath_db_image:", path_db_img[-22:])
#                print("score:", score)
#                print("matches:", matches[:4])
#        except cv2.error: # as err:
#            print("WARNING: Not enough keypoints found on db image.")
#            #print("error message: {0}".format(err))
#        except ZeroDivisionError:
#            print("WARNING: No matches between keypoints found.")
#        
#    if len(scores) <= 0:
#        print("No match found.")
#        return None, None
#        #return np.zeros((50,50,3)), None
#    #print(scores)
#    bestMatch, _ = getBestMatch(scores)
#    print("Match found with score:", bestMatch[1])
#    
#    if bestMatch[1] >= cf.matching_score_threshold:
#        print("Bad match, score too high.")
#        return bestMatch[0],bestMatch[1]
#    
#    return bestMatch[0], bestMatch[1]
