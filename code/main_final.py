# -*- coding: utf-8 -*-
"""
@author: Groep 5 PC visie 2020: Reiner, Maxime, Laurens, Elias
"""
import cv2
import numpy as np
import timeit
import config as cf
import painting_detection as pd
import matching as m
import file_tools as ft
from localization import main_localization

try:
    print("Press ctrl+C to abort.")
    for nr in range(16,17):
        #ONLY CHANGE THIS PARAMETER FOR DIFFERENT VIDEO
        videonummer = nr
#        save_csv = 0 #0 = do nothing, 1 = save db matches and score to csv file
        
        if videonummer < 10:
            videopad = "{}MSK_0{}.mp4".format(cf.path_smartphone, videonummer)
            undist_videopad = videopad
        elif videonummer < 12:
            videopad = "{}MSK_{}.mp4".format(cf.path_smartphone, videonummer)
            undist_videopad = videopad
        else:
            videopad = "{}MSK_{}.mp4".format(cf.path_gopro, videonummer)
            undist_videopad = videopad.replace(".mp4", "_undistorted.mp4")
        print("Video:",videopad)
        start_time = timeit.default_timer()
        list_db_match_paths, list_db_match_scores, list_kps_and_matches = [],[],[]
        
        #SAVING/LOADING DESCRIPTORS
        database = m.makeListFromDatabase(cf.path_paintings + "*")
        m.saveDescriptorFiles(database, cf.overwrite_descriptors)
        all_descriptors = m.loadDescriptorFiles(cf.path_descriptor_files + "*")
        
        #DETECTING SHARP FRAMES AND GETTING VIDEO PROPERTIES
        total_frames, fps, height, width = pd.getVideoProperties(videopad)
        _, frame_nrs, sharpness = pd.getSharpFrames(videopad, [0, ':'], 30)
        #frame_nrs = np.arange(0,total_frames, 500)
        blur_it = pd.calculateBlurIterations(height, width)
        
        #STARTING DETECTION, MATCHING & LOCALIZATION
        for i, frame_nr in enumerate(frame_nrs):
            #print("\nFrame nr.:", frame_nr)
            frame = m.getFrame(undist_videopad, frame_nr, False)
            extracted_paintings, list_poly_corners = m.extractPaintings(frame, frame_nr, blur_it)
            for j, painting in enumerate(extracted_paintings):
                db_match_path, db_match_score, kps_and_matches = m.findDatabaseMatch2(painting, database, all_descriptors)
                if db_match_path is not None:
                    list_db_match_paths.append(db_match_path[db_match_path.find("zaal_"):db_match_path.find("/Zaal")])
                    list_db_match_scores.append(db_match_score)
                    list_kps_and_matches.append(kps_and_matches)
                    inked_frame = m.inkFrame(frame, frame_nr, list_poly_corners[j], sharpness[i], db_match_score)
                    db_match_img = cv2.imread(db_match_path)
                    m.toonCanvas(inked_frame, painting, db_match_img, db_match_path, kps_and_matches, 1)
                        
            floorplan_img, rooms = main_localization(list_db_match_paths.copy(), list_db_match_scores.copy(), 
                                                     list_kps_and_matches.copy())
        
#        #SAVING FOUND MATCHES TO CSV FOR DEBUGGING
#        if save_csv:
#            ft.writeToCSV(videopad, list_db_match_paths, list_db_match_scores, 1)
        
        #ft.printDetectionsToFile(videonummer,frame_nrs,list_db_match_paths)
        print("Detected rooms:", rooms)
        print("\nRuntime:", str(round(timeit.default_timer() - start_time, 1)) + "s")
        print("DONE: Press any key to exit program")
        cv2.imshow("RESULT",floorplan_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("KeyboardInterrupt detected. Aborting...")
        raise SystemExit(0)
        