import os
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow import keras
from keras import backend as K
from unet import build_unet
from matplotlib import pyplot as plt

def jacard_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (intersection+1.0)/(K.sum(y_true)+K.sum(y_pred_f)-intersection+1.0)

import tensorflow as tf
def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(K.cast(y_true, dtype='float32'))
    #y_true_f=np.array(y_true_f, dtype=np.uint8)
    y_pred_f = K.flatten(K.cast(y_pred, dtype='float32'))
    #y_pred_f=np.array(y_pred_f, dtype=np.uint8)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred):
	gamma=2.0
	alpha=0.25
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

def total_loss(y_true, y_pred):
    cfe=focal_loss(y_true, y_pred)
    dl=dice_coef_loss(y_true, y_pred)
    return cfe+dl

class_labels=np.load("class_label.npy")

seed=24
batch_size=16
n_classes=8

from keras.utils import to_categorical
#preprocess_input=sm.get_preprocessing('resnet34')


def preprocess_data(img, mask, num_classes):
    def rgb_to_2D_label(label):
        label_seg=np.zeros(label.shape, dtype=np.uint8)
        for i, p in enumerate(class_labels,0):
            label_seg[np.all(label==p, axis=-1)]=i
        label_seg=label_seg[:,:,0]
        return label_seg
    img=img/255
    #img=tf.keras.applications.resnet.preprocess_input(img)
    labels=[]
    for i in range(mask.shape[0]):
        label=rgb_to_2D_label(mask[i])
        labels.append(label)
    labels=np.array(labels)
    labels=np.expand_dims(labels, axis=3)
    label_encoded=[]
    for i in labels:
        label_encoded.append(to_categorical([i],num_classes)[0])

    mask=np.array(label_encoded, dtype=np.uint8)
    return (img, mask)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
def trainGenerator (train_img_path, train_mask_path, num_class):
    img_data_gen_args = dict(horizontal_flip=True,
    vertical_flip=True, fill_mode='reflect')
    image_datagen = ImageDataGenerator (**img_data_gen_args)
    mask_datagen = ImageDataGenerator (**img_data_gen_args)
    image_generator = image_datagen.flow_from_directory( train_img_path, class_mode=None,
    batch_size = batch_size, seed=seed)
    mask_generator = mask_datagen.flow_from_directory(train_mask_path, class_mode=None,
     batch_size = batch_size, seed = seed)
    train_generator=zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask= preprocess_data(img, mask, num_class)
        yield (img, mask)


train_img_path="AIS_Dataset/train_images/"
train_mask_path="AIS_Dataset/train_masks/"
train_img_gen=trainGenerator(train_img_path, train_mask_path, num_class=8)
val_img_path="AIS_Dataset/val_images/"
val_mask_path="AIS_Dataset/val_masks/"
val_img_gen=trainGenerator(val_img_path, val_mask_path, num_class=8)
test_img_path="AIS_Dataset/test_images/"
test_mask_path="AIS_Dataset/test_masks/"
test_img_gen=trainGenerator(test_img_path, test_mask_path, num_class=8)

x,y=train_img_gen.__next__()

num_train_imgs=len(os.listdir('AIS_Dataset/train_images/train'))
num_val_images=len(os.listdir('AIS_Dataset/val_images/val'))
steps_per_epoch=num_train_imgs//8
val_steps_per_epoch=num_val_images//8
img_height=x.shape[1]
img_width=x.shape[2]
n_classes=8

checkpoint_filepath = "my_model_checkpoint.{epoch:02d}.weights.h5"

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch'
)

metrics=['accuracy', jacard_coef]
model=build_unet((256, 256, 3))
model.compile(optimizer='adam', loss=total_loss, metrics=metrics )

print(model.summary())

history=model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, epochs=50, verbose=1, validation_data=test_img_gen, validation_steps=val_steps_per_epoch, callbacks=[checkpoint_callback])