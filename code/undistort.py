# Undisort.py
# Created by Chris Rillahan
# Last Updated: 02/04/2015
# Written with Python 2.7.2, OpenCV 2.4.8 and NumPy 1.8.0

# This program takes a video file and removes the camera distortion based on the
# camera calibration parameters.  The filename and the calibration data filenames
# should be changed where appropriate.  Currently, the program is set to search for
# these files in the folder in which this file is located.

# This program first loads the calibration data.  Secondly, the video is loaded and
# the metadata is derived from the file.  The export parameters and file structure
# are then set-up.  The file then loops through each frame from the input video,
# undistorts the frame and then saves the resulting frame into the output video.
# It should be noted that the audio from the input file is not transfered to the
# output file.

import numpy as np
import cv2
import timeit
import sys
import config as cf
import glob

#Get all "distorted" video paths
filenames = glob.glob(cf.path_gopro+"MSK_??.mp4")[0:]
#Fix '\\' which sometimes occurs when using glob
for i in range(len(filenames)):
    filenames[i] = filenames[i].replace('\\', '/')
#print(filenames)

#filename = cf.path_gopro + "MSK_12.mp4"
calibration_data_fname = 'calibration_W_data.npz'

print('Loading data files')

npz_calib_file = np.load(calibration_data_fname)

distCoeff = npz_calib_file['distCoeff']
intrinsic_matrix = npz_calib_file['intrinsic_matrix']

npz_calib_file.close()

print('Finished loading files')
print(' ')
print('Starting to undistort the videos...')
for fn in filenames:
    print("\nNr:", str(filenames.index(fn)+1)+'/'+str(len(filenames)))
    print("Current video:", fn)
    # Opens the video import and sets parameters
    video = cv2.VideoCapture(fn)
    # Checks to see if a the video was properly imported
    status = video.isOpened()
    
    if status == True:
        codec = int(video.get(cv2.CAP_PROP_FOURCC))
        #codec = cv2.VideoWriter_fourcc('H', '2', '6', '4')
        FPS = video.get(cv2.CAP_PROP_FPS)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_lapse = (1 / FPS) * 1000
    
        # Initializes the export video file
        video_out = cv2.VideoWriter(str(fn[:-4]) + '_undistorted.mp4', codec, FPS, size, 1)
    
        # Initializes the frame counter
        current_frame = 0
        start = timeit.default_timer()
    
        while current_frame < total_frames:
            progress = (current_frame+1)*100/total_frames
            print("\rUndistorting video: [%-25s] %d%%" % ('♥'*int(progress/4), 
                  round(progress,0)),end="", flush=True)
            success, image = video.read()
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    
            dst = cv2.undistort(image, intrinsic_matrix, distCoeff, None)
    
            video_out.write(dst)
    
        video.release()
        video_out.release()
        duration = (timeit.default_timer() - float(start)) / 60
    
        print(' ')
        print('Finished undistorting the video')
        print('This video took: ' + str(round(duration,2)) + ' minutes')
    else:
        print("Error: Video failed to load")
        sys.exit()