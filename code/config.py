# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:55:23 2020

@author: Groep 5 2020
"""
#CONFIG FILE: contains path to database, videos and source files
#path_gopro = "D:/Users/Elias/Documents/data_project_computervisie/gopro/"
#path_smartphone = "D:/Users/Elias/Documents/data_project_computervisie/smartphone/"
#path_database = "D:/Users/Elias/Documents/data_project_computervisie/Computervisie 2020 Project Database/"
##smaller database used for the matching:
#path_paintings = "D:/Users/Elias/Documents/data_project_computervisie/db_downgraded/"
#path_calibration_images = "D:/Users/Elias/Documents/data_project_computervisie/calibration_images/"
#path_descriptor_files = "D:/Users/Elias/Documents/data_project_computervisie/db_downgraded_pickles/"
##database to extract smaller database from:
#path_paintings_high_res = "D:/Users/Elias/Documents/data_project_computervisie/hoge_resolutie_db/"

# REINER
# path_gopro = "D:/Computervisie 2020 Project Database/gopro/"
# path_smartphone = "D:/Computervisie 2020 Project Database/smartphone/"
# path_database = "D:/Computervisie 2020 Project Database/"
# path_paintings = "D:/Computervisie 2020 Project Database/db_downgraded/"
# path_calibration_images = "D:/Computervisie 2020 Project Database/calibration_images/"
# path_descriptor_files = "D:/Computervisie 2020 Project Database/db_downgraded_pickles/"

#MAXIME
#path_gopro = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/gopro/"
#path_smartphone = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/smartphone/"
#path_database = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/"
#path_paintings = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/db_downgraded/"
#path_calibration_images = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/calibration_images/"
#path_descriptor_files = "C:/Users/Maxime Carella/PycharmProjects/computervisie_project_groep5/code/project_data/db_downgraded_pickles/"

# Laurens
path_gopro = "videos/gopro/"
path_smartphone = "videos/smartphone/"
path_database = "not_used"
path_paintings = "db_downgraded/"
path_calibration_images = "not_used"
path_descriptor_files = "db_downgraded_pickles/"

# PAINTING DETECTION & MATCHING CONSTANTS: contains values for extracting paintings and matching
sharpness_threshold = 75  # Only use videoframes sharper (= bigger) than this value
# You should overwrite the previous descriptor files when changing value below!
MAX_FEATURES = 500  # Default: 500. Max keypoints to detect in an image.
max_hamming_distance = 40  # max hamming distance to mark 2 keypoints as a good match
remove_matches_with_score = 15 #Remove matches lower than this value. 0 = remove nothing. 15 is a descent value
blur_it_factor = 92160 # Height*width / this = #blur iterations. Results in 10 for gopro vids

# PLOT, PRINT and STORE OPTIONS:
overwrite_descriptors = 0  # Whether or not the descriptors of the db should be recalculated.
show_matches = 1 #Whether or not the matches between keypoints should be shown
show_frames = 1 #Whether or not the frames with detected paintings should be shown
show_extraction = 1 #Whether or not the thresholding and extraction should be shown
show_runtime_modules = 0 #Whether or not the runtime of the different modules should be shown