import numpy as np
import cv2
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from keras.models import load_model
from functions import *


if __name__ == '__main__':
    model = load_model('keras_mnist.h5')
    sum_list = []
    for i in range(0, 10):
        video_name = "{0}{1}{2}".format("video-", str(i), ".avi")
        video_path = "{0}{1}".format("videos/", video_name)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        x1, y1, x2, y2 = find_line(frame)
        digits = []
        sum_of_digits = 0
        while True:
            ret, frame = cap.read()
            if ret is not True:
                break

            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imageb = invert(image_bin(grayscale))
            # imageb = erode(dilate(img))
            image, regions = find_digits_for_prediction(frame, imageb, x1, y1, x2, y2, digits)

            for region in regions:
                digit = region.reshape(1, 1, 28, 28)
                prediction = model.predict(digit)
                found_digit = np.argmax(prediction)
                # print 'Nasao sam:', found_digit
                sum_of_digits += found_digit

            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        sum_list.append((video_name, sum_of_digits))
        print 'Suma brojeva za', video_name, "je", sum_of_digits

        cap.release()
        cv2.destroyAllWindows()

    print_to_file(sum_list)
