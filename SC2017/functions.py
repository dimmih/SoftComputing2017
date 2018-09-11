import numpy as np
import cv2
import math


def print_to_file(sum_list):
    with open('my_result.txt', 'w') as file:
        file.write('RA 200/2014 Dimitrije Mihajlovski\n')
        file.write('file	sum\n')
        for sum in sum_list:
            file.write("{0}{1}{2}{3}".format(sum[0], ' ', sum[1], '\n'))
        file.close()


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin_ret = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin_ret


def invert(image):
    return 255 - image


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


class Digit:
    def __init__(self, x, y, summed):
        self.x = x
        self.y = y
        self.summed = summed


def check_distance(x, y, digits):
    for digit in digits:
        distance = math.sqrt(math.pow(x - digit.x, 2) + math.pow(y - digit.y, 2))
        if distance < 15:
            return digit
    return None


def find_digits_for_prediction(image_orig, imagebin, x1, y1, x2, y2, digits):
    _, contours, _ = cv2.findContours(imagebin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    regions_array = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 15 and h > 13 and h < 50 and w > 3:
            digit = check_distance(x, y, digits)
            if digit is None:
                digit = Digit(x, y, False)
                digits.append(digit)
            else:
                digit.x = x
                digit.y = y
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if x >= x1 and x <= x2 and y <= y1 and y >= y2:
                if y - y1 >= ((y2 - y1) / (x2 - x1)) * (x - x1):
                    if digit.summed is False:
                        digit.summed = True
                        region = imagebin[y:y + h, x:x + w]
                        regions_array.append(resize_region(region))
                        # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image_orig, regions_array


def find_line(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask)

    # grayscale = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(result, 50, 200, None, 3)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)

    for x1, y1, x2, y2 in lines[0]:
        min_x = x1
        min_y = y1
        max_x = x2
        max_y = y2

    for i in range(len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if x1 < min_x:
            min_x = x1
            min_y = y1
        if x2 > max_x:
            max_x = x2
            max_y = y2

    cv2.line(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    # cv2.imshow('line', image)
    # print(min_x, min_y, max_x, max_y)

    return min_x, min_y, max_x, max_y
