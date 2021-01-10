import cv2
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from UNet import UNet

import tensorflow as tf
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

unet = UNet()
unet.load_weights(epoch=12150)
model = unet.get_model()

data_loader = DataLoader(None)
data_loader.evaluate_audio('./eval/My recording 11.mp3', model)

