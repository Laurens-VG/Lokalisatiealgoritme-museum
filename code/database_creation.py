import cv2
import numpy as np
import glob
import csv
import scipy.spatial.distance
import math
import os

point = ()
pt = ()
schilderij_teller = 0
original_img = []
resize_value = 0

'''
Choose a database in main at the end of the file. Doubleclick to select a painting. You can select multiple. If you
have more than one selection, split the image by pressing 'c'. Do this until every painting has its own window.
After this you can press 's' to save the image info to the csv database.
You can go to the next picture by pressing 'n'. You can reset to original by pressing 'r'.
'''


class Image:
    def __init__(self, image, number, rectangles):
        self.image = image
        self.name = "Image_" + str(number)
        self.rectangles = rectangles


# Main function to create the database. Takes all images from a room for processing.
def create_database_single_room(room):
    global point, original_img, resize_value
    path = "project_data/dataset_pictures_msk/zaal_" + room + "/*.jpg"
    start = 0
    resize_value = 6
    # start csv
    file = open('data_paintings.csv', 'w', newline='')
    header = 'Image_name Hall Corners'
    header = header.split()
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    while True:
        img_name = glob.glob(path)
        original_img = cv2.imread(img_name[start])
        (h, w) = original_img.shape[:2]
        print(h)
        img = ResizeWithAspectRatio(original_img, width=int(w / resize_value))
        index_IMG = img_name[start].find('IMG')
        extract_painting(img, room, img_name[start][index_IMG:])
        start += 1


# Iterative function that uses user input to extract the painting from an image.
def extract_painting(img, room, external_name):
    global point, pt, rect_for_tfo, schilderij_teller, original_img
    img0 = Image(img, 0, [])
    cv2.imshow(img0.name, img0.image)
    next = False
    point = (0, 0)
    pt = ()
    rect_for_tfo = []
    orig = img0.image.copy()
    all_imgs = [img0]
    print("Press c to crop, r to reset, n to get the next image, s to save, t to transfrom")
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("images/zaal_" + str(room)):
        os.makedirs("images/zaal_" + str(room))
    while next is False:
        for i in range(len(all_imgs)):
            cv2.imshow(all_imgs[i].name, all_imgs[i].image)
            cv2.setMouseCallback(all_imgs[i].name, get_points, all_imgs[i])
        key = cv2.waitKey()
        if key == ord("n"):
            schilderij_teller += 1
            next = True
        if key == ord("c"):
            accept_rect = all_imgs[i].rectangles
            copy = all_imgs[i].image
            print(accept_rect)
            for j in range(len(accept_rect)):
                x = accept_rect[j][0] + 5
                y = accept_rect[j][1] + 5
                w = accept_rect[j][2] - 7
                h = accept_rect[j][3] - 7
                new_img = copy[y:y + h, x:x + w]
                if j == 0:
                    all_imgs[i].image = new_img
                else:
                    new_class_image = Image(new_img, len(all_imgs) + i, [(x, y, w, h)])
                    all_imgs.append(new_class_image)
        if key == ord("r"):
            for k in range(len(all_imgs)):
                del all_imgs[0]
            cv2.destroyAllWindows()
            reset_img = Image(orig.copy(), 0, [])
            all_imgs.append(reset_img)
            rect_for_tfo = []
        if key == ord("s"):
            write_csv(external_name, room, all_imgs[0].rectangles)
            for k in range(len(all_imgs)):
                original_sized = get_original_boundaries(all_imgs[k])
                cv2.imwrite(
                    "images/zaal_" + str(room) + "/Zaal" + str(room) + "_schilderij" + str(schilderij_teller) + ".png",
                    original_sized)
                schilderij_teller += 1
            for l in range(len(all_imgs)):
                del all_imgs[0]
            cv2.destroyAllWindows()
            next = True
        if key == ord("t"):
            cv2.waitKey(1)
            for m in range(1, 5):
                print("Dubbelklik met de rechter muisknop op de hoek en druk vervolgens"
                      " op h om de volgende hoek in te geven.")
                cv2.setMouseCallback(all_imgs[i].name, get_points_for_transformation, all_imgs[i])
                cv2.waitKey(0)
                print("Dit was de " + str(m) + "e hoek op positie " + str(pt))
            rect_for_tfo = np.array(rect_for_tfo)
            rect = order_points(rect_for_tfo)
            warped = warpToRectangle(all_imgs[i].image, rect)
            warped_orig = warpToRectangle(original_img, rect * resize_value)
            new_class_image = Image(warped, len(all_imgs) + i, [])
            all_imgs[0].image = warped
            cv2.imwrite(
                "images/zaal_" + str(room) + "/Zaal" + str(room) + "_schilderij" + str(schilderij_teller) + ".png",
                warped_orig)
            schilderij_teller += 1
            next = True
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_original_boundaries(img_object):
    global original_img, resize_value
    print(img_object.rectangles)
    x, y, w, h = img_object.rectangles[0]
    x = x * resize_value
    y = y * resize_value
    w = w * resize_value
    h = h * resize_value
    resized_img = original_img[y:y + h, x:x + w]
    return resized_img


# Function to be called on mouse click
def get_points(event, x, y, flags, param):
    global point, pt
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point = (x, y)
        print(point)
        class_img = get_shape(param)
        cv2.imshow(class_img.name, class_img.image)


def get_points_for_transformation(event, x, y, flags, param):
    global point, pt, rect_for_tfo
    if event == cv2.EVENT_RBUTTONDBLCLK:
        pt = (x, y)
        rect_for_tfo.append(pt)


# Takes an image and finds a suited contour to be drawn
def get_shape(class_image):
    global point, pt
    img = class_image.image
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    kernel = np.ones((30, 30), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    erosian = cv2.erode(dilation, kernel, iterations=1)
    ret, thresh = cv2.threshold(erosian, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        if x < point[0] < x + w and y < point[1] < y + h:
            class_image.rectangles.append((x, y, w, h))
            cv2.rectangle(class_image.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return class_image


def order_points(pts):
    # initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[3] = pts[np.argmin(s)]
    rect[1] = pts[np.argmax(s)]
    # now, compute the difference between the points, the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[2] = pts[np.argmin(diff)]
    rect[0] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct the set of destination points to obtain a
    # "birds eye view", (i.e. top-down view) of the image, again specifying points  in the top-left, top-right,
    # bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def warpToRectangle(img, p):
    # image center
    h, w = img.shape[:2]
    u0 = (h) / 2.0
    v0 = (w) / 2.0
    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])
    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])
    w = max(w1, w2)
    h = max(h1, h2)
    # visible aspect ratio
    ar_vis = float(w) / float(h)
    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')
    # Points are differently sorted than input points: Fix this:
    temp = m3
    m3 = m4
    m4 = temp
    # calculate the focal disrance
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
    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
            n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))
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
    #        W = int(w)
    #        H = int((w/ar_real)/div)
    else:
        H = int(h)
        W = int(ar_real * H)
    #        H = int(h)
    #        W = int((ar_real*H)/div)
    # ADDED 4/04
    # W, H = int(W/3), int(H/3)
    pts1 = np.array(p).astype('float32')  # cornerpoints of unwarped image
    pts2 = np.float32([[0, H], [W, H], [W, 0], [0, 0]])  # cornerpoints of new warped image
    # print("pts needed: [[0,H],[W,H],[W,0],[0,0]]")
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (W, H))
    return dst


# Writes data to csv file
def write_csv(image_name, room, corners):
    to_append = f'{image_name} {room} {corners}'
    file = open('data_paintings.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


# functie om afbeelding te resizen om het volledig te kunnen zien op het scherm
# resizen gebeurt rekening houdend met de originele dimensie van de afbeelding
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


if __name__ == "__main__":
    # Runs the program for one particular set. Example: '1', '10', 'S', 'P'
    create_database_single_room("12")
