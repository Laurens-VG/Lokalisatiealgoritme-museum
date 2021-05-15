# -*- coding: utf-8 -*-
"""
Imports videofile and finds usefull frames (images) to send to 'painting_detection.py'
Created on Tue Apr  7 17:14:58 2020

@author: Elias, Reiner
"""
import config as cf
import numpy as np
import painting_detection as pd
import warping as w
import os
from sys import exit
import cv2
import pickle

def testVideoAllFramesWithSkipping(video_path, startframe, skipping):
    cap = cv2.VideoCapture(video_path)
    frame_nr = 0
    failed_frames = []
    succes_frames = []
    #i = 0
    i = 0
    cap.set(1, startframe)  #Added by elias
    frame_nr = startframe-1
    while cap.isOpened():
        i += 1
        frame_nr += 1
        ret, frame = cap.read()
        # input = frame.copy()
        print("Frame: " + str(frame_nr))
        if i % skipping == 0:
            i = 0
            cv2.putText(frame, "frame: " + str(frame_nr), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow("frame", frame)
            if frame_nr > startframe:
                try:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    #if pd.variance_of_laplacian(frame) >= cf.sharpness_threshold: #added
                    # print("\tsharpness_value:", variance_of_laplacian(frame))
                    
                    cntrs, _ = pd.getContours(frame, frame_gray, 10)
                    # poly_corners = pd.getLargestPolygon(frame, cntrs)
                    list_poly_corners = pd.getAllPolygons(frame, cntrs)
                    for i, poly_corners in enumerate(list_poly_corners, start=1):
                        for corner in poly_corners:
                            cv2.circle(frame, (corner[0][0], corner[0][1]), 5, (0, 0, 255), -1)
                        pd.origin = pd.getMiddlePoint(poly_corners)
                        res = w.warpToRectangle(frame, pd.sortCorners(poly_corners))
                        cv2.destroyWindow("res" + str(i))
                        cv2.imshow("res" + str(i), res)
                        exit(0)
                    cv2.imshow("frame", frame)
                    succes_frames.append(frame_nr)
                except ZeroDivisionError:
                    cv2.imshow("frame", frame)
                    print("ERROR: framenr " + str(frame_nr) + " aborted: Wrong polygon. (probaly floor)")
                    failed_frames.append(frame_nr)
                except IndexError:
                    cv2.imshow("frame", frame)
                    print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                    failed_frames.append(frame_nr)
                except OverflowError:
                    cv2.imshow("frame", frame)
                    print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                    failed_frames.append(frame_nr)
                except ValueError:
                    cv2.imshow("frame", frame)
                    print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                    failed_frames.append(frame_nr)
                except Exception:
                    print("ERROR: framenr " + str(frame_nr) + " aborted: An error ocurred")
                    failed_frames.append(frame_nr)
                    
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    return succes_frames, failed_frames


def testVideoAllFrames(startframe):
    video_path = cf.path_gopro + "MSK_12.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_nr = 0
    failed_frames = []
    succes_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # input = frame.copy()
        frame_nr += 1
        print("Frame: " + str(frame_nr))
        cv2.putText(frame, "frame: " + str(frame_nr), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow("frame", frame)
        if frame_nr > startframe:
            try:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cntrs, _ = pd.getContours(frame, frame_gray, 10)
                # poly_corners = pd.getLargestPolygon(frame, cntrs)
                list_poly_corners = pd.getAllPolygons(frame, cntrs)
                for i, poly_corners in enumerate(list_poly_corners, start=1):
                    for corner in poly_corners:
                        cv2.circle(frame, (corner[0][0], corner[0][1]), 5, (0, 0, 255), -1)
                    pd.origin = pd.getMiddlePoint(poly_corners)
                    res = w.warpToRectangle(frame, pd.sortCorners(poly_corners))
                    cv2.destroyWindow("res" + str(i))
                    cv2.imshow("res" + str(i), res)
                cv2.imshow("frame", frame)
                succes_frames.append(frame_nr)
            except ZeroDivisionError:
                cv2.imshow("frame", frame)
                print("ERROR: framenr " + str(frame_nr) + " aborted: Wrong polygon. (probaly floor)")
                failed_frames.append(frame_nr)
            except IndexError:
                cv2.imshow("frame", frame)
                print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                failed_frames.append(frame_nr)
            except OverflowError:
                cv2.imshow("frame", frame)
                print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                failed_frames.append(frame_nr)
            except ValueError:
                cv2.imshow("frame", frame)
                print("ERROR: framenr " + str(frame_nr) + " aborted: Not enough cornerpoints.")
                failed_frames.append(frame_nr)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def testVideo(video_path):
    # GETTING FRAMES FROM VIDEO AND EXTRACTING PAINTINGS
    frames, frame_nrs = pd.getSharpFrames(video_path, [0, 2000], step = 5)
    # pickle.dump(frames, open("frames_temp", "wb"))
    # pickle.dump(frame_nrs, open("frame_nrs_temp", "wb"))
    # frames = pickle.load(open("frames_temp", "rb"))
    # frame_nrs = pickle.load(open("frame_nrs_temp", "rb"))
    failed_images = []
    nr_of_images = len(frame_nrs)
    cnt = 0
    for i in range(0, len(frame_nrs)):  # Run through frames
        print("------------------")
        img_nr = frame_nrs[i]
        print("Framenr:", img_nr)
        try:
            img = pd.getImageFromVideo(video_path, img_nr)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            cntrs, _ = pd.getContours(img, img_gray, 10)
            poly_corners = pd.getLargestPolygon(img, cntrs)
            pd.origin = pd.getMiddlePoint(poly_corners)
            res = w.warpToRectangle(img, pd.sortCorners(poly_corners))
            pd.showImage(img, res, "input")
            cnt += 1
        except ZeroDivisionError:
            print("ERROR: framenr " + str(img_nr) + " aborted: Wrong polygon. (probaly floor)")
            failed_images.append(cnt)
        except IndexError:
            print("ERROR: framenr " + str(img_nr) + " aborted: Not enough cornerpoints.")
            failed_images.append(img_nr)
        except OverflowError:
            print("ERROR: framenr " + str(img_nr) + " aborted: Not enough cornerpoints.")
            failed_images.append(img_nr)
       # except Exception:
       #     print("ERROR: imagenr "+str(img_nr)+" failed.")
       #     failed_images.append(img_nr)

    print("\nEnd of script. Failed images:", failed_images)
    print("Performance, #succes/total_tried:", 100 * ((nr_of_images - len(failed_images)) / nr_of_images), '%')


def testPictures():
    # TEST THIS FIRST: THIS CODE USES THE TEST IMAGES 'test_pictures_msk'
    failed_images = []
    start, end = 0, 30
    nr_of_images = end - start
    src_directory = cf.path_database + "test_pictures_msk/"
    pathList = os.listdir(src_directory)  # Get list of all files

    for g in range(start, end):  # Run through images one by one
        print("------------------")
        print("Imagename:", pathList[g], "| Imagenr:", g)
        src_string = src_directory + pathList[g]
        try:
            img, img_gray = pd.readImage(src_string)
            height, width = img.shape[:2]
            # pd.origin = [0,int(width/2)]
            cntrs, _ = pd.getContours(img, img_gray)
            poly_corners = pd.getLargestPolygon(img, cntrs)
            pd.origin = pd.getMiddlePoint(poly_corners)
            print("centerpoint", pd.origin)
            res = w.warpToRectangle(img, pd.sortCorners(poly_corners))
            pd.showImage(img, res, name="input")
        except ZeroDivisionError:
            print("ERROR: imagenr " + str(g) + " aborted: Wrong polygon. (probaly floor)")
            failed_images.append(g)
        except IndexError:
            print("ERROR: imagenr " + str(g) + " aborted: Not enough cornerpoints.")
            failed_images.append(g)
    #    except Exception:
    #        print("ERROR: imagenr "+str(g)+" failed.")
    #        failed_images.append(g)

    print("\nEnd of script. Failed images:", failed_images)
    print("Performance, #succes/total_tried:", 100 * ((nr_of_images - len(failed_images)) / nr_of_images), '%')

def main():
    try:
        video_path = cf.path_gopro + "MSK_12.mp4"
        # testPictures()
        testVideo(video_path)
        # testVideoAllFrames(0)
        #video_path = cf.path_smartphone + "MSK_04.mp4"
        #succes_frames, failed_frames = testVideoAllFramesWithSkipping(video_path, 3208, 30)
        #print("\nsucces_frames", succes_frames)
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Aborting...")
        cv2.destroyAllWindows()
        exit(0)
    
if __name__ == "__main__":
    main()
