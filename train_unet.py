import os
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow import keras
from keras import backend as K
from unet import build_unet
from matplotlib import pyplot as plt

class_labels=np.load("class_label.npy")
batch_size = 16
seed = 42
n_classes = 8

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
    labels=np.expand_dims(labels, axis=3) ##modified axis=3 to axis=0
    label_encoded=[]
    for i in labels:
        label_encoded.append(to_categorical([i],num_classes)[0])

    mask=np.array(label_encoded, dtype=np.float32)
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

from matplotlib import pyplot as plt


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
a,b=val_img_gen.__next__()

num_train_imgs=len(os.listdir('AIS_Dataset/train_images/train'))
num_val_images=len(os.listdir('AIS_Dataset/val_images/val'))
steps_per_epoch=num_train_imgs//8
val_steps_per_epoch=num_val_images//8
img_height=x.shape[1]
img_width=x.shape[2]
n_classes=8

def jacard_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (intersection+1.0)/(K.sum(y_true)+K.sum(y_pred_f)-intersection+1.0)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K


def multi_unet_model(n_classes=8, IMG_HEIGHT=255, IMG_WIDTH=255, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #model.summary()

    return model

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm


weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]


metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
print(model.summary())


checkpoint_filepath = "my_model_checkpoint.{epoch:02d}.weights.h5"

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch'
)

history=model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, epochs=5, verbose=1, validation_data=val_img_gen, validation_steps=val_steps_per_epoch, callbacks=[checkpoint_callback])
#model.save('UAVid_AttUnet.hdf5')