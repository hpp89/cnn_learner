import cv2
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def display_image(file_path, delay=0):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    cv2.imshow('img', img)
    cv2.waitKey(delay)
