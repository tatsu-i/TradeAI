#---------------------------------------------------
# network
#---------------------------------------------------
import os
from keras.models import Sequential
from keras.layers import *

def build_compile_model(
        input_shape,         # 入力shape
        enable_image_layer,  # image_layerを入れるか
        window_length,       # window_length
        nb_actions,          # アクション数
        enable_dueling_network,  # dueling_network を有効にするか
        dense_units_num,         # Dense層のユニット数
        create_optimizer_func,
        metrics,                 # compile に渡す metrics
    ):
    model = Sequential()
    model.add(Flatten(input_shape=(1, ) + input_shape))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(nb_actions, activation='linear'))

    # compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer=create_optimizer_func(),
        metrics=metrics)

    return model
