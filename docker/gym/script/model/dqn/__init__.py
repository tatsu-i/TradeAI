import os
import json
import timeit
import warnings
import numpy as np
from glob import glob
from rl.callbacks import TrainEpisodeLogger
from rl.callbacks import TrainIntervalLogger
from keras.callbacks import Callback
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Input,
    concatenate,
    Dropout
)
from common.api import create_graph
from common.gymenv import Market

class TrainLogger(TrainEpisodeLogger):
    def __init__(self, model_name="default", model_group="best", visualize=True):
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.logs = []
        self.model_name = model_name
        self.model_group = model_group
        self.visualize = visualize
        self.step = 0
        self.best_metric = -99999999
        self.current_profit = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        self.rewards[episode] = []

    def on_episode_end(self, episode, logs):
        target = "sum reward"
        current_metric= np.sum(self.rewards[episode])
        #target = "net profit"
        #current_metric = self.current_profit
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            print("The reward is higher than the best one, saving checkpoint weights")
            base_path = os.environ.get("MODEL_PATH", "/data/model")
            model_path = os.path.join(base_path, self.model_group)
            newWeights = os.path.join(model_path, f"{self.model_name}.hdf5")
            self.model.save_weights(newWeights, overwrite=True)
            model = create_model(input_shape=(1,135), nb_actions=3)
            model.load_weights(newWeights)
            create_graph(self.model_name, self.model_group, model)
        else:
            print("The reward is lower than the best one, checkpoint weights not updated.")
        print(f"episode:{episode} best {target}: {self.best_metric}")

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.current_profit = logs['info']['profit']
        self.rewards[episode].append(logs['reward'])
        self.step += 1

def create_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(16, activation='softmax'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

def DQN(**kwargs):
    window_length = kwargs.get("window_length", 1)
    observation_space = kwargs["observation_space"]
    nb_actions = kwargs["nb_actions"]
    memory = SequentialMemory(limit=120000, window_length=window_length)
    policy = EpsGreedyQPolicy(eps=0.1)
    #policy = BoltzmannQPolicy()
    kwargs["model"] = create_model(
        input_shape=(window_length,) + observation_space,
        nb_actions=nb_actions
    )
    kwargs["policy"] = policy
    kwargs["memory"] = memory
    del kwargs["observation_space"]
    del kwargs["window_length"]
    dqn_agent = DQNAgent(**kwargs)
    dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn_agent

import os
import pickle
from retry import retry
from keras.optimizers import Adam
import tensorflow as tf

@retry(tries=3, delay=2, backoff=1)
def train_dqn(csvpath, model_name, model_group, nb_steps, num_actors=6, mode="stg"):
    base_path="/data"
    nb_actions = 3
    csvfiles = glob(csvpath)
    env = Market(csvfiles, nb_actions, debug=False, mode=mode)
    model_path = os.path.join(base_path, "model", model_group)
    np.random.seed(369)
    env.seed(369)
    train_callback= TrainLogger(model_name=model_name, model_group=model_group, visualize=False)
    # 引数が多いので辞書で定義して渡しています。
    args={
        'observation_space': env.observation_space.shape,
        'nb_actions': env.action_space.n,
        'nb_steps_warmup': 300,
        'window_length': 1,
        'batch_size': 1024,
        'gamma': 0.6,
        #'train_interval': 1024,
        'enable_double_dqn': True
    }
    agent = DQN(**args)
    agent.compile(optimizer=Adam())
    weight_file = os.path.join(model_path, f"{model_name}.hdf5")
    if os.path.exists(weight_file):
        print(f"load best weight {weight_file}")
        agent.model.load_weights(weight_file)
    print(agent.model.summary())
    # 訓練
    with tf.device("/device:GPU:0"):
        history = agent.fit(
            env,
            nb_steps=nb_steps,
            visualize=False,
            verbose=0,
            callbacks=[
                train_callback,
            ]
        )
