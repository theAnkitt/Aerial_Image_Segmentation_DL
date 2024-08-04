import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, BatchNormalization  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model


def conv_block(x, num_filters):
    x=Conv2D(num_filters, 3, padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.2)(x)
    x=Conv2D(num_filters, 3, padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    return x
    

def Encoder_block(x, num_filters):
    x=conv_block(x, num_filters)
    p=MaxPooling2D((2,2))(x)
    return x,p

def decoder_block(x, s, num_filters):
    x=Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(x)
    x=Concatenate()([x,s])
    x=conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs=Input(input_shape)

    #Encoder
    s1,p1=Encoder_block(inputs, 16)
    s2,p2=Encoder_block(p1, 32)
    s3,p3=Encoder_block(p2,64)
    s4,p4=Encoder_block(p3,128)
    x=conv_block(p4, 256)

    #Decoder
    d1=decoder_block(x, s4, 128)
    d2=decoder_block(d1, s3, 64)
    d3=decoder_block(d2, s2, 32)
    d4=decoder_block(d3, s1, 16)

    outputs=Conv2D(8, 1, padding='same', activation='softmax')(d4)

    model=Model(inputs, outputs, name="Unet")
    return model