# -*- coding: utf-8 -*-
import os

def writeToCSV(videopad, list_db_match_paths, list_db_match_scores, met_sharpness):
    opslaan_pad = videopad.split('/')
    naam_bestand = opslaan_pad[-1].replace(".mp4","")
    if not os.path.exists('./test_output/'):
        os.makedirs('./test_output/')
    if met_sharpness == 0:
        ofile = open('./test_output/{}met_sharpness.csv'.format(naam_bestand), 'w')
    else:
        ofile = open('./test_output/{}zonder_sharpness.csv'.format(naam_bestand), 'w')
    ofile.write('match_zaal;score\n')
    for i, elem in enumerate(list_db_match_paths):
        naam_zaal = elem.split('/')[-1]
        ofile.write(naam_zaal+';'+str(list_db_match_scores[i])+'\n')
    ofile.close()
    return 0

def readFromCSV(video_nr, met_sharpness, sharp_thresh):
    if met_sharpness == 0:
        path = "./test_output_{}/MSK_{}zonder_sharpness.csv".format(sharp_thresh,video_nr)
    else:
        path = "./test_output/MSK_{}met_sharpness.csv".format(video_nr)
    path = "./test_output_sharpness15/MSK_09zonder_sharpness.csv"
    
    list_db_match_paths = []
    list_db_match_scores = []
    try:
        with open(path) as f:
            next(f)
            for line in f:
                line = line.strip('\n')
                naam_zaal, score = line.split(';')[:]
                list_db_match_paths.append(naam_zaal)
                list_db_match_scores.append(int(score))
    except:
        print("Error: reading csv failed.")
#    print("paths:", list_db_match_paths)
#    print("scores:", list_db_match_scores)
    return list_db_match_paths, list_db_match_scores
        