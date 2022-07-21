#---------------------------------------------------
# actor
#---------------------------------------------------
import os
import numpy as np
import random
import time
import traceback
import rl.core
from .network import build_compile_model

from collections import deque
class Actor(rl.core.Agent):
    def __init__(self,
        actor_index,
        model_args,
        args,
        experience_q,
        model_sync_q,
        **kwargs):
        super(Actor, self).__init__(**kwargs)

        self.actor_index = actor_index
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q

        self.nb_actions = args["nb_actions"]
        self.input_shape = args["input_shape"]
        self.window_length = args["window_length"]
        self.actor_model_sync_interval = args["actor_model_sync_interval"]
        self.gamma = args["gamma"]
        self.epsilon = args["epsilon"]
        self.multireward_steps = args["multireward_steps"]
        self.action_interval = args["action_interval"]

        # local memory
        self.local_memory = deque()
        self.local_memory_update_size = args["local_memory_update_size"]

        self.learner_train_num = 0

        # reset
        self.reset_states()

        # model
        self.model = build_compile_model(**model_args)
        self.compiled = True

    def reset_states(self):
        self.recent_action = 0
        self.repeated_action = 0
        self.recent_reward = [0 for _ in range(self.multireward_steps)]
        self.recent_observations = [np.zeros(self.input_shape) for _ in range(self.window_length+self.multireward_steps)]

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        pass

    def forward(self, observation):
        self.recent_observations.append(observation)  # 最後に追加
        self.recent_observations.pop(0)  # 先頭を削除

        if self.training:
            # 結果を送る
            # multi step learning、nstepの報酬を割引しつつ加算する
            reward = 0
            for i, r in enumerate(reversed(self.recent_reward)):
                reward += r * (self.gamma ** i)

            state0 = self.recent_observations[:self.window_length]
            state1 = self.recent_observations[-self.window_length:]

            # priority のために TD-error をだす。
            state0_qvals = self.model.predict(np.asarray([state0]), 1)[0]
            state1_qvals = self.model.predict(np.asarray([state1]), 1)[0]

            maxq = np.max(state1_qvals)

            td_error = reward + self.gamma * maxq
            td_error = state0_qvals[self.recent_action] - td_error

            # local memoryに追加
            self.local_memory.append((state0, self.recent_action, reward, state1, td_error))


        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:

            # 行動を決定
            if self.training:

                # ϵ-greedy法
                if self.epsilon > np.random.uniform(0, 1):
                    # ランダム
                    action = np.random.randint(0, self.nb_actions)
                else:
                    # model入力用にwindow長さ分取得
                    obs = np.asarray([self.recent_observations[-self.window_length:]])

                    # 最大のQ値を取得
                    q_values = self.model.predict(obs, batch_size=1)[0]
                    action = np.argmax(q_values)
            else:
                # model入力用にwindow長さ分取得
                obs = np.asarray([self.recent_observations[-self.window_length:]])

                # Q値が最大のもの
                q_values = self.model.predict(obs, batch_size=1)[0]
                action = np.argmax(q_values)

            # リピート用
            self.repeated_action = action

        # backword用に保存
        self.recent_action = action

        return action


    def backward(self, reward, terminal):
        self.recent_reward.append(reward)  # 最後に追加
        self.recent_reward.pop(0)  # 先頭を削除

        if self.training:
            # 一定間隔で model を learner からsyncさせる
            if self.step % self.actor_model_sync_interval == 0:
                # 要求を送る
                self.model_sync_q[0].put(1)  # 要求

            # weightが届いていれば更新
            if not self.model_sync_q[1].empty():
                weights = self.model_sync_q[1].get(timeout=1)
                # 空にする(念のため)
                while not self.model_sync_q[1].empty():
                    self.model_sync_q[1].get(timeout=1)
                self.model.set_weights(weights)

            # localメモリが一定量超えていれば RemoteMemory に送信
            if len(self.local_memory) > self.local_memory_update_size:
                # 共有qに送るのは重いので配列化
                data = []
                while len(self.local_memory) > 0:
                    data.append(self.local_memory.pop())
                self.experience_q.put(data)

        return []

    @property
    def layers(self):
        return self.model.layers[:]

def actor_run(
    actor_index,
    actor_func,
    model_args,
    args,
    experience_q,
    model_sync_q,
    logger_q,
    actors_end_q,
    ):
    #print("Actor{} Starts!".format(actor_index))
    try:
        actor = Actor(
            actor_index,
            model_args,
            args,
            experience_q,
            model_sync_q,
        )

        # model load
        if os.path.isfile( args["load_weights_path"] ):
            actor.model.load_weights(args["load_weights_path"])

        # run
        actor_func(actor_index, actor)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        #print("Actor{} End!".format(actor_index))
        actors_end_q.put(1)
