import os
import cv2
import math
from tqdm import tqdm
import numpy as np

import utils

# out_dir = "F:\\tianchi\\test\\result\images"
# out_dir = "F:\\tianchi\\FaceRecognitionAdversarial\\FLM\\images"
out_dir = "F:\\tianchi\\test\\result\images_712_24_6"
origin_dir = "F:\\tianchi\FaceRecognitionAdversarial\data\securityAI_round1_images"


def evaluate_difference(out_image, origin_image):
    # image_name = dirs[index]
    # out_image = cv2.imread(os.path.join(out_dir, image_name))
    # origin_image = cv2.imread(os.path.join(origin_dir, image_name))

    """遍历图像每个像素的每个通道"""
    # print(origin_image.shape)  # 打印图像的高，宽，通道数（返回一个3元素的tuple）
    height = origin_image.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
    width = origin_image.shape[1]
    channels = origin_image.shape[2]
    # print("height:%s,width:%s,channels:%s" % (height, width, channels))
    # print(origin_image.size)  # 打印图像数组内总的元素数目（总数=高X宽X通道数）
    errors = 0
    for row in range(height):  # 遍历每一行
        for col in range(width):  # 遍历每一列
            error_pixel = 0
            for channel in range(channels):  # 遍历每个通道（三个通道分别是BGR）
                sub_pixel = abs(origin_image[row][col][channel] - out_image[row][col][channel])
                if sub_pixel <= 25.5:
                    error_pixel += sub_pixel * sub_pixel
                elif origin_image[row][col][channel] - out_image[row][col][channel] > 25.5:
                    out_image[row][col][channel] = origin_image[row][col][channel] - 25.5
                    error_pixel += 25.5 * 25.5
                elif out_image[row][col][channel] - origin_image[row][col][channel] > 25.5:
                    out_image[row][col][channel] = origin_image[row][col][channel] + 25.5
                    error_pixel += 25.5 * 25.5
            errors += math.sqrt(error_pixel)
            error_average = errors / (height * width)
    print("每幅图平均差异：%f" % error_average)
    return errors / (height * width) ,out_image


if __name__ == "__main__":
    dirs = os.listdir(out_dir)
    average_error = 0

    for index in tqdm(range(len(dirs))):
        image_name = dirs[index]
        out_image = utils.read_img(os.path.join(out_dir, image_name))

        origin_image = utils.read_img(os.path.join(origin_dir, image_name))
        errors,out_image = evaluate_difference(out_image.astype(np.int), origin_image.astype(np.int))
        # utils.show_image(out_image)
        # utils.show_difference(out_image,origin_image)

        average_error += errors
    average_error = average_error / (len(dirs))
    print(average_error)
