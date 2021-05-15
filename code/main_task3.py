# -*- coding: utf-8 -*-
"""
Imports videofile and finds usefull frames (images) to send to 'painting_detection.py'
Created on Tue Apr  7 17:14:58 2020

@author: Elias, Reiner
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
import config as cf
import painting_detection as pd
import matching as m
import warping as w
from localization import main_localization


def main_realtime(videopad):
    start_time = timeit.default_timer()
    all_scores = []
    good_frame = False
    success_frames = 0
    correct_matches = 0
    false_matches = 0

    database = m.makeListFromDatabase(cf.path_paintings + "*")
    m.saveDescriptorFiles(database, cf.overwrite_descriptors)
    all_descriptors = m.loadDescriptorFiles(cf.path_descriptor_files + "*")

    total_frames, fps, height, width = pd.getVideoProperties(videopad)
    # Getting frame_nrs from video and returning useable sharp frames.
    _, frame_nrs, sharpness = pd.getSharpFrames(videopad, [0, ':'], 60)  # step 60 for checking frames
    # frame_nrs = np.arange(0,total_frames, 60)
    sharpness = np.zeros(len(frame_nrs))
    # blur_it = pd.calculateBlurIterations(height, width)

    list_db_match_paths = []
    list_db_match_scores = []
    list_kps_and_matches = []
    i = 0
    for frame_nr in range(0, total_frames, 30):  # step 30 (real_time)
        print("\n{}".format('*' * 40), "\nFrame nr.:", frame_nr)
        # voor gopro
        undist_videopad = videopad.replace(".mp4", "_undistorted.mp4")
        # voor smartphone
        # undist_videopad = videopad
        frame = m.getFrame(undist_videopad, frame_nr, False)
        cv2.imshow("video", frame)
        cv2.waitKey(1)
        if frame_nr in frame_nrs:
            extracted_paintings, list_poly_corners = m.extractPaintings(frame, frame_nr)
            for j, painting in enumerate(extracted_paintings):
                print("\nPolygon:", j + 1)
                print("----------------------------------------")
                db_match_path, db_match_score, kps_and_matches = m.findDatabaseMatch2(painting, database,
                                                                                      all_descriptors)
                if db_match_path is not None:
                    list_db_match_paths.append(db_match_path[db_match_path.find("zaal_"):db_match_path.find("/Zaal")])
                    list_db_match_scores.append(db_match_score)
                    list_kps_and_matches.append(kps_and_matches)
                    inked_frame = m.inkFrame(frame, frame_nr, list_poly_corners[j], sharpness[i], db_match_score)
                    db_match_img = cv2.imread(db_match_path)
                    if m.toonCanvas(inked_frame, painting, db_match_img, db_match_path, kps_and_matches, 1) == 'y':
                        correct_matches += 1
                    else:
                        false_matches += 1
                    all_scores.append(db_match_score)
                    if db_match_score is not None and db_match_score > 0:
                        good_frame = True
            if good_frame:
                success_frames += 1
                good_frame = False
        floorplan_img = main_localization(list_db_match_paths.copy(), list_db_match_scores.copy(),
                                          list_kps_and_matches.copy())
        cv2.imshow("floorplan", floorplan_img)
        cv2.waitKey(1)
    #    print("\nall scores:")
    #    print(all_scores)
    return 0

    m.calcPerformance(success_frames, len(frame_nrs), correct_matches, false_matches)
    print("Runtime:", str(round(timeit.default_timer() - start_time, 1)) + "s")


def main_noSharpness(videopad, step=10):
    start_time = timeit.default_timer()
    all_scores = []
    good_frame = False
    success_frames = 0
    correct_matches = 0
    false_matches = 0

    database = m.makeListFromDatabase(cf.path_paintings + "*")
    m.saveDescriptorFiles(database, cf.overwrite_descriptors)
    all_descriptors = m.loadDescriptorFiles(cf.path_descriptor_files + "*")

    total_frames, fps, height, width = pd.getVideoProperties(videopad)
    # Getting frame_nrs from video and returning useable sharp frames.
    # _, frame_nrs, sharpness = pd.getSharpFrames(videopad, [0, ':'], 60)  # step 60 for checking frames
    # frame_nrs = np.arange(0,total_frames, 60)
    # sharpness = np.zeros(len(frame_nrs))
    # blur_it = pd.calculateBlurIterations(height, width)

    list_db_match_paths = []
    list_db_match_scores = []
    list_kps_and_matches = []
    i = 0
    for frame_nr in range(0, total_frames, step):
        print("\n{}".format('*' * 40), "\nFrame nr.:", frame_nr)
        # voor gopro
        undist_videopad = videopad.replace(".mp4", "_undistorted.mp4")
        # voor smartphone
        # undist_videopad = videopad
        frame = m.getFrame(undist_videopad, frame_nr, False)
        # if frame_nr in frame_nrs:
        extracted_paintings, list_poly_corners = m.extractPaintings(frame, frame_nr)
        for j, painting in enumerate(extracted_paintings):
            print("\nPolygon:", j + 1)
            print("----------------------------------------")
            db_match_path, db_match_score, kps_and_matches = m.findDatabaseMatch2(painting, database,
                                                                                  all_descriptors)
            if db_match_path is not None:
                list_db_match_paths.append(db_match_path[db_match_path.find("zaal_"):db_match_path.find("/Zaal")])
                list_db_match_scores.append(db_match_score)
                list_kps_and_matches.append(kps_and_matches)

                inked_frame = m.inkFrame(frame, frame_nr, list_poly_corners[j], 0, db_match_score)
                db_match_img = cv2.imread(db_match_path)
                if m.toonCanvas(inked_frame, painting, db_match_img, db_match_path, kps_and_matches, 1) == 'y':
                    correct_matches += 1
                else:
                    false_matches += 1

                all_scores.append(db_match_score)
                if db_match_score is not None and db_match_score > 0:
                    good_frame = True
        if good_frame:
            success_frames += 1
            good_frame = False
        floorplan_img = main_localization(list_db_match_paths.copy(), list_db_match_scores.copy(),
                                          list_kps_and_matches.copy())
    #    print("\nall scores:")
    #    print(all_scores)

    opslaan_pad = videopad.split('/')
    naam_bestand = opslaan_pad[-1].replace(".mp4", "")
    if not os.path.exists('./test_output/'):
        os.makedirs('./test_output/')
    ofile = open('./test_output/' + naam_bestand + "met_sharpness" + ".csv", "w")
    ofile.write('match_zaal;score\n')
    for i, elem in enumerate(list_db_match_paths):
        naam_zaal = elem.split('/')[-1]
        ofile.write(naam_zaal + ';' + str(list_db_match_scores[i]) + '\n')
    ofile.close()

    # m.calcPerformance(success_frames, len(frame_nrs), correct_matches, false_matches)
    print("Runtime:", str(round(timeit.default_timer() - start_time, 1)) + "s")


def main_elias(videopad, showPictures=True):
    start_time = timeit.default_timer()
    all_scores = []
    good_frame = False
    success_frames = 0
    correct_matches = 0
    false_matches = 0

    database = m.makeListFromDatabase(cf.path_paintings + "*")
    m.saveDescriptorFiles(database, cf.overwrite_descriptors)
    all_descriptors = m.loadDescriptorFiles(cf.path_descriptor_files + "*")

    total_frames, fps, height, width = pd.getVideoProperties(videopad)
    # Getting frame_nrs from video and returning useable sharp frames.
    _, frame_nrs, sharpness = pd.getSharpFrames(videopad, [0, ':'], 60)
    # frame_nrs = np.arange(0,total_frames, 60)
    # sharpness = np.zeros(len(frame_nrs))
    # blur_it = pd.calculateBlurIterations(height, width)

    list_db_match_paths = []
    list_db_match_scores = []
    list_kps_and_matches = []

    for i, frame_nr in enumerate(frame_nrs):
        print("\n{}".format('*' * 40), "\nFrame nr.:", frame_nr)
        # voor gopro
        undist_videopad = videopad.replace(".mp4", "_undistorted.mp4")
        # voor smartphone
        # undist_videopad = videopad

        frame = m.getFrame(undist_videopad, frame_nr, False)
        extracted_paintings, list_poly_corners = m.extractPaintings(frame, frame_nr)
        for j, painting in enumerate(extracted_paintings):
            print("\nPolygon:", j + 1)
            print("----------------------------------------")
            db_match_path, db_match_score, kps_and_matches = m.findDatabaseMatch2(painting, database, all_descriptors)
            if db_match_path is not None:
                list_db_match_paths.append(db_match_path[db_match_path.find("zaal_"):db_match_path.find("/Zaal")])
                list_db_match_scores.append(db_match_score)
                list_kps_and_matches.append(kps_and_matches)
                if showPictures:
                    inked_frame = m.inkFrame(frame, frame_nr, list_poly_corners[j], sharpness[i], db_match_score)
                    db_match_img = cv2.imread(db_match_path)
                    if m.toonCanvas(inked_frame, painting, db_match_img, db_match_path, kps_and_matches) == 'y':
                        correct_matches += 1
                    else:
                        false_matches += 1
                all_scores.append(db_match_score)
                if db_match_score is not None and db_match_score > 0:
                    good_frame = True
        if good_frame:
            success_frames += 1
            good_frame = False
        floorplan_img, rooms = main_localization(list_db_match_paths.copy(), list_db_match_scores.copy(),
                                          list_kps_and_matches.copy())
    #    print("\nall scores:")
    #    print(all_scores)

    opslaan_pad = videopad.split('/')
    naam_bestand = opslaan_pad[-1].replace(".mp4", "")
    if not os.path.exists('./test_output/'):
        os.makedirs('./test_output/')
    ofile = open('./test_output/' + naam_bestand + "zonder_sharpness" + ".csv", "w")
    ofile.write('match_zaal;score\n')
    for i, elem in enumerate(list_db_match_paths):
        naam_zaal = elem.split('/')[-1]
        ofile.write(naam_zaal + ';' + str(list_db_match_scores[i]) + '\n')
    ofile.close()

    print("Runtime:", str(round(timeit.default_timer() - start_time, 1)) + "s")


def makeSmallerDatabase():  # Downgrade database and store smaller resolution images
    database = m.makeListFromDatabase(cf.path_paintings_high_res + "*")
    m.downgradeDatabase(database)


if __name__ == "__main__":
    try:
        # main_elias(cf.path_gopro + "MSK_18.mp4", False) #14 gaat het filmpje niet door alle zalen
        # main_realtime(cf.path_gopro + "MSK_15.mp4", True)
        for i in range(12, 20):
            main_noSharpness(cf.path_gopro + "MSK_{}.mp4".format(i), 10)
        # main_reiner()
    except KeyboardInterrupt:
        # cv2.destroyAllWindows()
        print("KeyboardInterrupt detected. Aborting...")
        raise SystemExit(0)
