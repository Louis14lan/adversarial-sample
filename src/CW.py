import numpy as np
import keras
from keras import backend
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications import vgg16
import cleverhans.attacks
import scipy.misc
import os
from tqdm import tqdm

keras_model = vgg16.VGG16(weights='imagenet')
# keras_model = inception_v3.InceptionV3(weights='imagenet')

input_size = 224

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess = backend.get_session()

IMAGE_SIZE = 112
embedding_SIZE = 512
IMAGE_DIR = "./images/"
RESULT_DIR = "./result/"
EPSILON = 16


dirs = os.listdir(IMAGE_DIR)
for index in tqdm(range(len(dirs))):
    image_name = dirs[index]
    image = scipy.misc.imread(os.path.join(IMAGE_DIR, image_name))

    # Resizing the image to be of size input_size * input_size
    image = np.array(scipy.misc.imresize(image, (input_size, input_size)),
                     dtype=np.float32)

    # converting each pixel to the range [0,1] (Normalization)
    image = np.array([image / 255.0])

    wrap = KerasModelWrapper(keras_model)

    # attack_method = attacks.SpatialTransformationMethod(wrap, sess = sess)
    attack_method = cleverhans.attacks.CarliniWagnerL2(wrap, sess = sess)

    # carlini and wagner
    attack_method_params = {'batch_size': 1,
                 'confidence': 10,
                 'learning_rate': 0.1,
                 'binary_search_steps': 5,
                 'max_iterations': 1000,
                 'abort_early': True,
                 'initial_const': 0.01,
                 'clip_min': 0,
                 'clip_max': 1}

    adv_cw = attack_method.generate_np(image)
    # adv_cw = attack_method.generate_np(image, **attack_method_params)
    scipy.misc.imsave(os.path.join(RESULT_DIR, image_name),scipy.misc.imresize(adv_cw[0], (112, 112)))