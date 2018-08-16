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


def read_resize_image(file_path, image_size):

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    s = 0
    if image.shape[0] < image.shape[1]:
        s = (int(round(float(image.shape[0]) * image_size) / image.shape[1]), image_size)
    else:
        s = (image_size, int(round(float(image.shape[1]) * image_size) / image.shape[0]))

    image = cv2.resize(image, (s[1], s[0]), interpolation=cv2.INTER_CUBIC)
    image = cv2.copyMakeBorder(image, 0, image_size - image.shape[0], 0, image_size - image.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image[:, :, ::-1]
