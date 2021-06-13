import numpy as np
from PIL import Image
import PIL.Image
from PIL import ImageEnhance
from datetime import timedelta, date

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

import cv2
import easyocr

# For I/O
import io
from pathlib import Path
import time
from datetime import  date
import re
import glob

# For logging
import logging

img_super_resolution_model = tf.keras.models.load_model("models/models/esrgan-tf2_1")

# Load classes classifier
classes_model = tf.keras.models.load_model('checkpoint/object_vgg16_model.h5')


def load_img(images_path):
    start = time.time()
    tensor_images = []
    classes_list = []
    image_path_list = []

    for image in glob.glob(images_path + "/*/*.jpg"):

        image_path = str(image).split('/', 1)[1]
        image_path_list.append(image_path)
        # classify classes
        class_pred = predict_classes(image)

        classes_list.append(class_pred)

        img = tf.io.read_file(str(image))
        # Decode a JPEG-encoded image to a uint8 tensor.
        img = tf.image.decode_jpeg(img, channels=3)

        tensor_images.append(img)
    logging.info("Time Taken in load_img : %f" % (time.time() - start))
    return tensor_images, image_path_list, classes_list


def get_image_tensor(img_path):
    return tf.image.decode_jpeg(img_path, channels=3)


def normalize(input_image, input_mask):
    start = time.time()
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    logging.info("Time Taken to normalize: %f" % (time.time() - start))
    return input_image, input_mask


def image_super_resolution(model, blur_image):
    start = time.time()

    fake_image = model(blur_image)
    fake_image = tf.squeeze(fake_image)

    logging.info("Time Taken to convert to high resolution: %f" % (time.time() - start))
    return fake_image


def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    start = time.time()
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    logging.info("Time Taken in convert_rgb_to_names: %f" % (time.time() - start))
    return names[index]


def load_image_train(datapoint):
    start = time.time()

    input_image = tf.image.resize(datapoint['image'], (244, 512))
    # position from where to extract the text
    input_mask = tf.image.crop_to_bounding_box(datapoint['segmentation_mask'], 400, 400, 60, 200)

    input_mask = tfa.image.sharpness(input_mask, 0.9)
    high_resolution_mask = image_super_resolution(img_super_resolution_model,
                                                  np.expand_dims(tf.dtypes.cast(input_mask, tf.float32), axis=0))

    input_image, input_mask = normalize(input_image, input_mask)
    logging.info("Time Taken to get input_image, input_mask, high_resolution_mask in load_image_train : %f" % (time.time() - start))
    return input_image, input_mask, high_resolution_mask


def display(display_list):
    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'True Mask', 'High Resolution Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def display_all_images(display_labeled_dict, num_of_images=3, start_from=0):
    plt.figure(figsize=(10, 10))
    print(f'start_from = {start_from}')
    for i in range(num_of_images):
        plt.subplot(int(num_of_images / 3), 3, i + 1)
        plt.title(display_labeled_dict[start_from]['text'])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_labeled_dict[start_from]['original_image']))
        plt.axis('off')
        start_from += 1
    plt.show()


def clean_text(text):
    start = time.time()
    # pattern check for AAA1234567
    pattern = re.compile(r'\s+|\.|\/|\,|\+|\"|\~|\#|\:|\?|\)')
    text_no_special_char = re.sub(pattern, '', text)

    if re.match(r'^[A-Z]{3}[0-9]{7}', text_no_special_char):
        print("Pattern matches...")
        text_code = text_no_special_char[:3].upper()
        text_number = text_no_special_char[3:10]
    else:
        print('Pattern does not match...')
        text_code = text_no_special_char[:3].upper()
        pattern2 = re.compile(r'[^0-9]')
        text_number_intermediate = re.sub(pattern2, '', text_no_special_char[3:])
        text_number = text_number_intermediate[:7]
    logging.info("Time Taken in clean_text: %f" % (time.time() - start))
    return text_code + text_number


def predict_classes(in_image_path):
    start = time.time()
    true_classes = {0: '10', 1: '100', 2: '20', 3: '5', 4: '50'}
    img = image.load_img(in_image_path, target_size=(224, 224))
    in_image_array = image.img_to_array(img)
    in_image_array = tf.expand_dims(in_image_array, 0)

    in_image_array = preprocess_input(in_image_array)

    pred = classes_model.predict(in_image_array)
    max_pred = np.argmax(pred, axis=1).tolist()
    str_pred = ''.join([str(v) for v in max_pred])
    logging.info("Time Taken in predict_classes: %f" % (time.time() - start))
    return true_classes[int(str_pred)]


if __name__ == "__main__":
    start_time = time.perf_counter()
    logging.basicConfig(filename="perf_log_" + date.today().strftime("%d%m%Y") + ".log", level=logging.INFO)
    in_image, img_path_list, class_list = load_img("jpegs/")
    labeled_images = []
    num_of_images = 0
    # print(in_image)
    for i in range(len(in_image) - 1):

        lbl_img = {}
        logging.info(i)
        print(in_image[i].shape)
        datapoint = {'image': in_image[i], 'segmentation_mask': in_image[i]}
        norm_tensor, text_tensor, high_resolution_tensor = load_image_train(datapoint)

        text_image = Image.fromarray(tf.dtypes.cast(high_resolution_tensor, tf.uint8).numpy())
        text_image.thumbnail(size=(244, 244))

        pil_image_sno = tf.keras.preprocessing.image.array_to_img(text_image)

        pil_image_norm = tf.keras.preprocessing.image.array_to_img(norm_tensor)

        rgb_value = pil_image_norm.getpixel((500, 224))
        rgb_color = convert_rgb_to_names(rgb_value)
        open_cv_image = np.array(pil_image_sno)

        easyocr_start = time.time()
        reader = easyocr.Reader(['en'])
        text = reader.readtext(open_cv_image, detail=0)
        print(text)
        logging.info("Time Taken to get text from easyocr in main loop : %f" % (time.time() - easyocr_start))

        predicted_classes = class_list[i]

        lbl_img['original_image'] = norm_tensor
        lbl_img['image_path'] = img_path_list[i]
        lbl_img['text'] = clean_text(''.join(text))

        print(lbl_img['text'])
        lbl_img['classes'] = predicted_classes
        print(f'classes = {lbl_img["classes"]}')
        lbl_img['rgb_val'] = '(' + ', '.join(str(val) for val in rgb_value) + ')'
        lbl_img['rgb_color'] = rgb_color

        labeled_images.append(lbl_img)
        num_of_images = i + 1

        ###display([norm_tensor, text_tensor, high_resolution_tensor])

    end_time = time.perf_counter()
    logging.info(f'Time taken to complete process {num_of_images} images and save to ES is {round(end_time - start_time, 2)} second(s)')


    ###display_all_images(labeled_images, 18, 0)
