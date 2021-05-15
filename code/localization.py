import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import file_tools as ft
from collections import OrderedDict
#from dtaidistance import dtw

global point


def main_localization(paths, scores, frames):
    #    print("-----------LOCALIZATION---------------")
    #    print(paths)
    #    print(scores)
    #    print(frames)
    img, rooms = visualization_correction(paths)
    return img, rooms



def get_neighbours(room):
    neighbours_temp = []
    neighbours = []
    with open('floorplan/adjacent_rooms.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            # print(row)
            if row[0] == room:
                neighbours_temp.append(row[1])
    # print("neighbours temp:")
    # print(neighbours_temp)
    if neighbours_temp:
        neighbours_temp[0] = neighbours_temp[0].split(',')
        for i in range(len(neighbours_temp[0])):
            neighbours.append(neighbours_temp[0][i])
    return neighbours


def get_neighbours_of_neigbours(room):
    neighbours_of_neighbours = []
    res = []
    neighbours = get_neighbours(room)
    for i in range(len(neighbours)):
        neighbours_of_neighbours.append(get_neighbours(neighbours[i]))
    for j in range(len(neighbours_of_neighbours)):
        for k in range(len(neighbours_of_neighbours[j])):
            res.append(neighbours_of_neighbours[j][k])
    return res


def remove_from_strings(rooms):
    for i in range(len(rooms)):
        rooms[i] = rooms[i][5:]


def visualization(rooms):
    img_path = "floorplan/floorplan_img.png"
    img = cv2.imread(img_path)
    remove_from_strings(rooms)
    common_neighbour_room = []
    with open('floorplan/location_rooms.csv') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        all_rows = []
        for rows in csv_reader1:
            all_rows.append(rows)
    for i in range(len(rooms) - 1):
        room1 = "iets"
        room2 = "iets"
        for rows in all_rows:
            # print(rows[0])
            if rows[0] == rooms[i]:
                room1 = rows
            if (rows[0] == rooms[i + 1]):
                room2 = rows
        cv2.line(img, (int(room1[1]), int(room1[2])), (int(room2[1]), int(room2[2])), (0, 0, 255),
                 thickness=3)
        cv2.line(img, (int(room1[1]), int(room1[2])), (int(room2[1]), int(room2[2])), (0, 0, 255),
                 thickness=3)
    # cv2.imshow("floorplan", img)
    # cv2.waitKey()
    return img


def visualization_correction(rooms):
    img_path = "floorplan/floorplan_img.png"
    image2 = cv2.imread(img_path)
    remove_from_strings(rooms)

    rooms = keep_double_matches(rooms)

    remove_not_neigbhours(rooms)
    #    print("1")
    #    print(rooms)

    common_neighbour_room = []
    with open('floorplan/location_rooms.csv') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        all_rows = []
        for rows in csv_reader1:
            all_rows.append(rows)
    with open('floorplan/location_passages.csv') as csv_file2:
        csv_reader2 = csv.reader(csv_file2, delimiter=',')
        all_rows_passages = []
        for rows_passages in csv_reader2:
            all_rows_passages.append(rows_passages)
    for i in range(len(rooms) - 1):
        # image2 = img.copy()
        room1 = "iets"
        room2 = "iets"
        passage = ""
        for rows in all_rows:
            if rows[0] == rooms[i]:
                room1 = rows
            if (rows[0] == rooms[i + 1]):
                room2 = rows
        for rows_passages in all_rows_passages:
            if rows_passages[0].find(room1[0]) != -1 and rows_passages[0].find(room2[0]) != -1:
                passage = rows_passages
            if room1[0] == "6" and room2[0] == "8":
                passage = ""
        if passage:  # er is een rechtstreekse verbinding tussen de kamers
            if room1 == room2:
                cv2.circle(image2, (int(room1[1]), int(room1[2])), 2, (0, 255, 0), thickness=3)
            else:
                cv2.line(image2, (int(room1[1]), int(room1[2])), (int(passage[1]), int(passage[2])), (0, 255, 0),
                         thickness=3)
                cv2.line(image2, (int(passage[1]), int(passage[2])), (int(room2[1]), int(room2[2])), (0, 255, 0),
                         thickness=3)
            # print("teken rechtstreekse verbinding tussen: ")
            # print(room1[0])
            # print(room2[0])
        else:  # er is geen rechtsrteekse verbinding tussen de kamers

            neighbours_room1 = get_neighbours(room1[0])
            neighbours_room2 = get_neighbours(room2[0])
            # print(neighbours_room1[0])
            # print(neighbours_room2[0])
            gevonden = 0
            # zoek gemeenschappelijke buur
            for l in range(len(neighbours_room1)):
                # print(neighbours_room1[l])
                for j in range(len(neighbours_room2)):
                    # print(neighbours_room2[j])
                    if neighbours_room1[l] != ',' and neighbours_room2[j] == neighbours_room1[l]:
                        # ze hebben een gemeenschappelijke buur
                        # print(common_neighbour_room) #fout
                        common_neighbour_room = neighbours_room1[l]
                        # print(common_neighbour_room)
                        gevonden = 1
            if gevonden == 0:
                continue
            # zoek coordinaten van gemeenschappelijke buur
            print(common_neighbour_room)
            # common_neighbour_room = int(common_neighbour_room)
            for rows in all_rows:
                if rows[0] == common_neighbour_room:
                    common_neighbour_room = rows
            # in common neighbour kamer met cordinaten
            print(common_neighbour_room)
            for rows_passages in all_rows_passages:
                if rows_passages[0].find(room1[0]) != -1 and rows_passages[0].find(common_neighbour_room[0]) != -1:
                    passage = rows_passages
                    print(passage)
            if passage:  # er is een rechtstreekse verbinding tussen de kamers
                cv2.line(image2, (int(room1[1]), int(room1[2])), (int(passage[1]), int(passage[2])), (0, 255, 0),
                         thickness=3)
                cv2.line(image2, (int(passage[1]), int(passage[2])),
                         (int(common_neighbour_room[1]), int(common_neighbour_room[2])), (0, 255, 0), thickness=3)
            for rows_passages in all_rows_passages:
                if rows_passages[0].find(room2[0]) != -1 and rows_passages[0].find(common_neighbour_room[0]) != -1:
                    passage = rows_passages
            if passage:  # er is een rechtstreekse verbinding tussen de kamers
                cv2.line(image2, (int(room2[1]), int(room2[2])), (int(passage[1]), int(passage[2])), (0, 255, 0),
                         thickness=3)
                cv2.line(image2, (int(passage[1]), int(passage[2])),
                         (int(common_neighbour_room[1]), int(common_neighbour_room[2])), (0, 255, 0), thickness=3)
            # print("teken niet rechtstreekse verbinding")
            # print(room1[0])
            # print(room2[0])
    rooms = remove_duplicates(rooms)
    return image2, rooms


def remove_duplicates(rooms):
    rooms = list(OrderedDict((x, True) for x in rooms).keys())
    return rooms


def get_location_rooms():
    # start csv
    file = open('floorplan/location_rooms.csv', 'w', newline='')
    header = 'room X Y'
    header = header.split()
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    img_path = "floorplan/floorplan_img.png"
    img = cv2.imread(img_path)
    cv2.imshow("floorplan", img)
    cv2.setMouseCallback("floorplan", get_point)
    # cv2.waitKey(0)


def get_location_passages():
    # start csv
    file = open('floorplan/location_passages.csv', 'w', newline='')
    header = 'passage between rooms A...Z X Y'
    header = header.split()
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    img_path = "floorplan/floorplan_img.png"
    img = cv2.imread(img_path)
    cv2.imshow("floorplan", img)
    cv2.setMouseCallback("floorplan", get_point)
    # cv2.waitKey(0)


def get_point(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point = (x, y)
        print(point)
        print("Between which rooms?")
        z = input()
        write_data_to_csv(z, point[0], point[1])


def write_data_to_csv(room, point1, point2):
    to_append = f'{room} {point1} {point2}'
    file = open('floorplan/location_passages.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


# def remove_unique_room(rooms):
#     for i in range(len(rooms) - 1, 0, -1):
#         if rooms.count(rooms[i]) == 1:
#             del rooms[i]

def remove_not_neigbhours(rooms):
    # remove_from_strings(rooms)
    if rooms:
        current_room = rooms[0]
        number_of_rooms_start = len(rooms)
        neighbours_current_room = get_neighbours(current_room)
        neigbours_of_neighbours_current_room = get_neighbours_of_neigbours(current_room)
        for i in range(len(rooms) - 1):
            i = i + len(rooms) - number_of_rooms_start
            neighbours_current_room = get_neighbours(rooms[i])
            neigbours_of_neighbours_current_room = get_neighbours_of_neigbours(rooms[i])
            if rooms[i + 1] not in neighbours_current_room and rooms[i + 1] not in neigbours_of_neighbours_current_room:
                del rooms[i + 1]


def keep_double_matches(rooms):
    res = []
    for i in range(len(rooms) - 1):
        if rooms[i] == rooms[i + 1]:
            res.append(rooms[i])
    return res

def show_individual_room(room):
    #remove_from_strings(room)
    room = room[5:]
    # print("room:")
    # print(room)
    # print(type(room))

    img_path = "floorplan/floorplan_img.png"
    img = cv2.imread(img_path)
    with open('floorplan/location_rooms.csv') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        all_rows = []
        for rows in csv_reader1:
        #     all_rows.append(rows)
        # for rows in all_rows:
        #     print(rows[0])
        #     print(room)
            # print(rows[0])
            if rows[0] == room:
                cv2.circle(img, (int(rows[1]), int(rows[2])), 20, (0, 255, 0), thickness=3)
    
    cv2.namedWindow("floorplan",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("floorplan", 923, 615) 
    cv2.imshow("floorplan", img)
    #cv2.waitKey(0)

if __name__ == "__main__":
    # print(getNeighbours("A"))
    # get_location_rooms()
    # get_location_passages()
    rooms = ["zaal_A", "zaal_B", "zaal_C", "zaal_D", "zaal_E", "zaal_F",
             "zaal_G", "zaal_H", "zaal_I", "zaal_J", "zaal_K", "zaal_L"]
    for room in rooms:
        show_individual_room(room)

    rooms_gopro_msk12 = ['zaal_15', 'zaal_A', 'zaal_19', 'zaal_A', 'zaal_A', 'zaal_A', 'zaal_A', 'zaal_B', 'zaal_B',
                         'zaal_B', 'zaal_B', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_15',
                         'zaal_D', 'zaal_G', 'zaal_G', 'zaal_F', 'zaal_15', 'zaal_F', 'zaal_G', 'zaal_G', 'zaal_G',
                         'zaal_G', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D']
    rooms_gopro_msk13 = ['zaal_O', 'zaal_N', 'zaal_N', 'zaal_N', 'zaal_N', 'zaal_N', 'zaal_N', 'zaal_19', 'zaal_H',
                         'zaal_H', 'zaal_H', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D',
                         'zaal_D', 'zaal_D', 'zaal_6', 'zaal_D', 'zaal_15', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D',
                         'zaal_D', 'zaal_D', 'zaal_D', 'zaal_15', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_15', 'zaal_D',
                         'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_15', 'zaal_D', 'zaal_D', 'zaal_D',
                         'zaal_15', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D',
                         'zaal_D', 'zaal_D', 'zaal_D', 'zaal_D', 'zaal_S', 'zaal_B', 'zaal_B', 'zaal_B']
    rooms_gopro_msk15 = ['zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S',
                         'zaal_S', 'zaal_S', 'zaal_S', 'zaal_B', 'zaal_15', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_F',
                         'zaal_15', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S',
                         'zaal_S', 'zaal_F', 'zaal_F', 'zaal_F', 'zaal_14', 'zaal_S', 'zaal_B', 'zaal_S', 'zaal_B',
                         'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S', 'zaal_S',
                         'zaal_V', 'zaal_V', 'zaal_19', 'zaal_V', 'zaal_V', 'zaal_V', 'zaal_V', 'zaal_S', 'zaal_18',
                         'zaal_15', 'zaal_R', 'zaal_15', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_S', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_S', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_S', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_15', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_S', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_15', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_15', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_15', 'zaal_6', 'zaal_19', 'zaal_19', 'zaal_15',
                         'zaal_19', 'zaal_15', 'zaal_19', 'zaal_19', 'zaal_11', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19',
                         'zaal_19', 'zaal_19', 'zaal_S', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_15',
                         'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_19', 'zaal_15', 'zaal_19', 'zaal_19',
                         'zaal_L', 'zaal_18', 'zaal_19', 'zaal_19', 'zaal_18', 'zaal_18', 'zaal_S', 'zaal_18',
                         'zaal_18', 'zaal_S', 'zaal_18', 'zaal_18', 'zaal_18', 'zaal_18', 'zaal_18', 'zaal_18',
                         'zaal_18', 'zaal_18', 'zaal_18', 'zaal_18', 'zaal_19', 'zaal_18', 'zaal_18', 'zaal_18',
                         'zaal_17', 'zaal_17', 'zaal_17', 'zaal_15']
    rooms_gopro_msk16 = ['zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_15', 'zaal_13', 'zaal_13',
                         'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13',
                         'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13',
                         'zaal_13', 'zaal_13', 'zaal_13', 'zaal_13', 'zaal_15', 'zaal_15', 'zaal_S', 'zaal_15',
                         'zaal_15', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8',
                         'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8',
                         'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8', 'zaal_8',
                         'zaal_8', 'zaal_8', 'zaal_8', 'zaal_7', 'zaal_7', 'zaal_7', 'zaal_7']
    rooms_gopro_msk14 = ['zaal_G', 'zaal_G', 'zaal_G', 'zaal_G', 'zaal_G', 'zaal_B', 'zaal_E', 'zaal_I', 'zaal_L',
                         'zaal_I', 'zaal_L',
                         'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I',
                         'zaal_I', 'zaal_I',
                         'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I',
                         'zaal_I', 'zaal_I',
                         'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I',
                         'zaal_I', 'zaal_I',
                         'zaal_L', 'zaal_L', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_I', 'zaal_10', 'zaal_L', 'zaal_10',
                         'zaal_J', 'zaal_J',
                         'zaal_L', 'zaal_L', 'zaal_J', 'zaal_L', 'zaal_10', 'zaal_10', 'zaal_10', 'zaal_10', 'zaal_K',
                         'zaal_K', 'zaal_K',
                         'zaal_K', 'zaal_I', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_I', 'zaal_L', 'zaal_L', 'zaal_7',
                         'zaal_L', 'zaal_L',
                         'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L',
                         'zaal_L', 'zaal_L',
                         'zaal_L', 'zaal_L', 'zaal_10', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L',
                         'zaal_L', 'zaal_L',
                         'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_L', 'zaal_11', 'zaal_11', 'zaal_10', 'zaal_11',
                         'zaal_11', 'zaal_11',
                         'zaal_L', 'zaal_K', 'zaal_10', 'zaal_9', 'zaal_9', 'zaal_9', 'zaal_9', 'zaal_9', 'zaal_9',
                         'zaal_K']
    rooms_gopro_msk14_met_score = [['zaal_G', 25], ['zaal_G', 38], ['zaal_G', 23], ['zaal_G', 21], ['zaal_B', 41],
                                   ['zaal_E', 17],
                                   ['zaal_I', 68], ['zaal_L', 21], ['zaal_I', 63], ['zaal_L', 28], ['zaal_I', 62],
                                   ['zaal_I', 37], ['zaal_I', 59], ['zaal_I', 25],
                                   ['zaal_I', 24], ['zaal_I', 66], ['zaal_I', 27], ['zaal_I', 60], ['zaal_L', 34],
                                   ['zaal_I', 72], ['zaal_I', 86], ['zaal_I', 66],
                                   ['zaal_I', 79], ['zaal_I', 55], ['zaal_I', 70], ['zaal_I', 55], ['zaal_I', 59],
                                   ['zaal_I', 44], ['zaal_I', 50], ['zaal_I', 39],
                                   ['zaal_I', 109], ['zaal_I', 50], ['zaal_I', 113], ['zaal_I', 58], ['zaal_I', 108],
                                   ['zaal_I', 53], ['zaal_I', 120], ['zaal_I', 55],
                                   ['zaal_I', 80], ['zaal_I', 45], ['zaal_I', 59], ['zaal_I', 39], ['zaal_I', 58],
                                   ['zaal_I', 25], ['zaal_L', 16], ['zaal_I', 18],
                                   ['zaal_I', 29], ['zaal_I', 19], ['zaal_I', 36], ['zaal_L', 23], ['zaal_L', 22],
                                   ['zaal_L', 17], ['zaal_J', 16], ['zaal_10', 25],
                                   ['zaal_J', 23], ['zaal_L', 17], ['zaal_C', 16], ['zaal_K', 18], ['zaal_J', 24],
                                   ['zaal_L', 17], ['zaal_K', 27], ['zaal_10', 26],
                                   ['zaal_10', 18], ['zaal_10', 16], ['zaal_L', 17], ['zaal_L', 20], ['zaal_K', 17],
                                   ['zaal_K', 22], ['zaal_K', 16], ['zaal_L', 40],
                                   ['zaal_L', 22], ['zaal_K', 18], ['zaal_I', 28], ['zaal_L', 26], ['zaal_L', 23],
                                   ['zaal_L', 24], ['zaal_J', 18], ['zaal_L', 26],
                                   ['zaal_L', 17], ['zaal_7', 25], ['zaal_L', 17], ['zaal_L', 29], ['zaal_L', 31],
                                   ['zaal_L', 17], ['zaal_L', 22], ['zaal_L', 48],
                                   ['zaal_I', 36], ['zaal_L', 17], ['zaal_L', 37], ['zaal_L', 20], ['zaal_L', 29],
                                   ['zaal_L', 21], ['zaal_L', 36], ['zaal_10', 30],
                                   ['zaal_L', 40], ['zaal_10', 25], ['zaal_L', 22], ['zaal_L', 20], ['zaal_L', 24],
                                   ['zaal_L', 27], ['zaal_L', 24], ['zaal_L', 20],
                                   ['zaal_L', 19], ['zaal_10', 24], ['zaal_K', 24], ['zaal_L', 42], ['zaal_L', 29],
                                   ['zaal_K', 18], ['zaal_L', 19], ['zaal_L', 18],
                                   ['zaal_11', 19], ['zaal_11', 64], ['zaal_11', 73], ['zaal_L', 16], ['zaal_K', 20],
                                   ['zaal_11', 28], ['zaal_11', 50], ['zaal_11', 40],
                                   ['zaal_K', 34], ['zaal_C', 16], ['zaal_I', 21], ['zaal_I', 18], ['zaal_I', 20],
                                   ['zaal_10', 20], ['zaal_9', 17], ['zaal_9', 27],
                                   ['zaal_9', 22], ['zaal_9', 30], ['zaal_9', 34]]


    # remove_from_strings(rooms)
    # print(rooms[0])
    # cv2.imshow('individueel',show_individual_room(rooms[0]))

    # a = np.array([[[1,3],[5,9],[6,8]],[[4,3],[5,1],[6,6]]])
    # print(a)
    #
    # # print(np.max(a[:,:,:]),6)
    #
    #
    # s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    # s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    # distance = dtw.distance(s1, s2)
    # print(distance)
    # rooms_msk, rooms_scores = ft.readFromCSV(1, 1, 25)
    # # rooms_gopro_msk17 =
    # # remove_from_strings(rooms_gopro_msk15)
    # # print(get_neighbours_of_neigbours('A'))
    # # remove_unique_room(rooms_gopro_msk15)
    # # print(rooms_gopro_msk14)
    # # visualization(rooms_gopro_msk14)
    # # print(rooms_gopro_msk13)
    # # remove_not_neigbhours(rooms_gopro_msk15)
    # # print(rooms_gopro_msk15)
    # # # print(rooms_gopro_msk13)
    #
    # # visualization_correction(rooms_gopro_msk15)
    # # rooms_gopro_msk15 = keep_double_matches(rooms_gopro_msk15)
    # print(rooms_msk)
    # img_route, rooms = visualization_correction(rooms_msk)
    # cv2.imshow("floorplan", img_route)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # visualization(rooms_gopro_msk15)
