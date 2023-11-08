import tensorflow as tf

from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.models import Model

def VGGNet_19(input_shape):
    X_input = Input(shape=input_shape)

    # conv_block_1
    X = Conv2D (filters=64, kernel_size=3, strides=(1,1), padding='same')(X_input)
    X = Activation('relu')(X)
    X = Conv2D (filters=64, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)

    # maxpool_block_1
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # conv_block_2
    X = Conv2D (filters=128, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=128, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)

    # maxpool_block_2
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # conv_block_3
    X = Conv2D (filters=256, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=256, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=256, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=256, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)

    # maxpool_block_3
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # conv_block_4
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)

    # maxpool_block_4
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # conv_block_5
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)
    X = Conv2D (filters=512, kernel_size=3, strides=(1,1), padding='same')(X)
    X = Activation('relu')(X)

    # maxpool_block_5
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # flatten_block
    X = Flatten()(X)

    # fc_block
    X = Dense(units=4096)(X)
    X = Activation('relu')(X)
    X = Dense(units=4096)(X)
    X = Activation('relu')(X)
    X = Dense(units=1000)(X)
    X_output = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X_output)

    return model

model = VGGNet_19(input_shape=(224,224,3))
print(model.summary())