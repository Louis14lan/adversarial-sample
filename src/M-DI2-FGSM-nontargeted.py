#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:
import argparse
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd
from evalute import evaluate_difference
from utils import read_img, write_img, distance_between_img

IMAGE_SIZE = 112
embedding_SIZE = 512
IMAGE_DIR = "./images/"
RESULT_DIR = "./result/"
EPSILON = 16


class Model():

    def __init__(self):
        from models import inception_resnet_v1  # facenet model
        self.network = inception_resnet_v1

        self.image_batch = tf.placeholder(tf.uint8, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='images')

        image = (tf.to_float(self.image_batch) - 127.5) / 128.0
        prelogits, _ = self.network.inference(image, 1.0, False, bottleneck_layer_size=embedding_SIZE)
        self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, './models/20180408-102900/model-20180408-102900.ckpt-90')  # facenet weight.

    def compute_victim(self, lfw_IMAGE_SIZE_path, name):
        imgfolder = os.path.join(lfw_IMAGE_SIZE_path, name)
        assert os.path.isdir(imgfolder), imgfolder
        images = glob.glob(os.path.join(imgfolder, '*.png')) + glob.glob(os.path.join(imgfolder, '*.jpg'))
        image_batch = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1] for f in images]
        for img in image_batch:
            assert img.shape[0] == IMAGE_SIZE and img.shape[1] == IMAGE_SIZE, \
                "--data should only contain IMAGE_SIZExIMAGE_SIZE images. Please read the README carefully."
        embeddings = self.eval_embeddings(image_batch)
        self.victim_embeddings = embeddings
        return embeddings

    def structure(self, input_tensor):
        """
        Args:
            input_tensor: NHWC
        """
        # create random angle rotation's adversarial sample
        # angle = tf.random_uniform((), -np.pi / 8, np.pi / 8, dtype=tf.float32)
        # rotated_image = tf.contrib.image.rotate(input_tensor, angle)

        rnd = tf.random_uniform((), 100, IMAGE_SIZE, dtype=tf.int32)
        rescaled = tf.image.resize_images(
            input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        h_rem = IMAGE_SIZE - rnd
        w_rem = IMAGE_SIZE - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
            pad_left, pad_right], [0, 0]])
        padded.set_shape((input_tensor.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
        output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)

        # # Adding Gaussian noise
        # noise = tf.random_normal(shape=tf.shape(output), mean=0.0, stddev=1.0,
        #                          dtype=tf.float32)
        # output = tf.add(output, noise)

        return output

    def build_pgd_attack(self, eps):
        victim_embeddings = tf.constant(self.victim_embeddings, dtype=tf.float32)

        def one_step_attack(image, grad):
            """
            core components of this attack are:
            (a) PGD adversarial attack (https://arxiv.org/pdf/1706.06083.pdf)
            (b) momentum (https://arxiv.org/pdf/1710.06081.pdf)
            (c) input diversity (https://arxiv.org/pdf/1803.06978.pdf)
            """
            orig_image = image
            image = self.structure(image)
            image = (image - 127.5) / 128.0
            image = image + tf.random_uniform(tf.shape(image), minval=-1e-2, maxval=1e-2)
            prelogits, _ = self.network.inference(image, 1.0, False, bottleneck_layer_size=embedding_SIZE)
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            objective = tf.sqrt(tf.reduce_sum(tf.square(embeddings - victim_embeddings), 1))
            noise, = tf.gradients(objective, image)

            noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
            noise = 0.9 * grad + noise

            adv = tf.clip_by_value(orig_image + tf.sign(noise), lower_bound, upper_bound)
            return adv, noise

        input = tf.to_float(self.image_batch)
        lower_bound = tf.clip_by_value(input - eps, 0, 255.)
        upper_bound = tf.clip_by_value(input + eps, 0, 255.)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            adv, _ = tf.while_loop(
                lambda _, __: True, one_step_attack,
                (input, tf.zeros_like(input)),
                back_prop=False,
                maximum_iterations=1000,
                parallel_iterations=1)
        self.adv_image = adv
        return adv

    def set_victim_embeddings(self, victim_embeddings):
        self.victim_embeddings = victim_embeddings

    def eval_attack(self, img):
        # img: single HWC image
        out = self.sess.run(
            self.adv_image, feed_dict={self.image_batch: [img]})[0]
        return out

    def eval_embeddings(self, batch_arr):
        return self.sess.run(self.embeddings, feed_dict={self.image_batch: batch_arr})

    def distance_to_victim(self, img):
        emb = self.eval_embeddings([img])
        dist = np.dot(emb, self.victim_embeddings.T).flatten()
        dist_mean = np.mean(self.victim_embeddings, axis=0)
        print(np.linalg.norm(emb - dist_mean))
        stats = np.percentile(dist, [10, 30, 50, 70, 90])
        return stats


if __name__ == '__main__':
    model = Model()
    dirs = os.listdir(IMAGE_DIR)
    for index in range(len(dirs)):
        image_name = dirs[index]
        print("正处理第%d张图片..." % index)
        origin_img = read_img(os.path.join(IMAGE_DIR, image_name))  # 读入原图片
        # show_image(origin_img)

        # 计算原图片embedding
        origin_emb = model.eval_embeddings([origin_img])
        model.set_victim_embeddings(origin_emb)
        model.build_pgd_attack(EPSILON)

        # 生成对抗样本
        out = model.eval_attack(origin_img).astype(np.int)
        # show_image(out)

        # 比较原图片与输出图片差异
        print("embedding之间差距为：%f" % distance_between_img(model, origin_img, out))
        evaluate_difference(out, origin_img)
        # show_difference(out,origin_img)

        # 保存对抗样本
        write_img(os.path.join(RESULT_DIR, image_name), out)
