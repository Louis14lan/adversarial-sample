import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2


def read_img(path):
    return cv2.imread(path)[:, :, ::-1].astype(np.int)

def write_img(path,img):
    cv2.imwrite(path, img[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def show_difference(image,origin_image):
    difference_image = np.abs(image-origin_image)
    # print(np.min(difference_image))
    difference_image_show = ((difference_image-np.min(difference_image))/(np.max(difference_image)-np.min(difference_image)))*255
    show_image(difference_image_show.astype(np.int))


def show_image(image):
    plt.figure()
    plt.imshow(image.astype(np.int))
    plt.show()



def convert_to_origin(image):
    image_show = ((image + 1.0) * 0.5 * 255)
    return cv2.resize(image_show, (112, 112))

def distance_between_img(model,out_image,origin_image):
    origin_emb = model.eval_embeddings([origin_image])
    out_emb = model.eval_embeddings([out_image])
    dst = distance_between_emb(origin_emb,out_emb)
    return dst

def distance_between_emb(origin_emb,out_emb):
    dst = np.sum(np.abs(out_emb-origin_emb))
    return dst