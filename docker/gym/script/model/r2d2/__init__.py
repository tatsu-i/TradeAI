import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない
import numpy as np
import random
import time
import traceback
import math
from glob import glob

import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import multiprocessing as mp
from common.api import create_graph
from common.gymenv import Market

#--- GPU設定
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto(
#    gpu_options=tf.GPUOptions(
#        #visible_device_list="0", # specify GPU number
#        allow_growth=True,
#        per_process_gpu_memory_fraction=1,
#    )
#)
#set_session(tf.Session(config=config))

#---------------------------------------------------
# manager
#---------------------------------------------------

class R2D2Manager():
    def __init__(self, 
        actor_func, 
        num_actors,
        args,
        create_optimizer_func,
        ):

        # type チェック
        lstm_types = [
            "",
            "lstm",
            "lstm_ful",
        ]
        if args["lstm_type"] not in lstm_types:
            raise ValueError('lstm_type is ["","lstm","lstm_ful"]')

        # lstm_ful のみburnin有効
        if args["lstm_type"] != "lstm_ful":
            args["burnin_length"] = 0
        

        self.actor_func = actor_func
        self.num_actors = num_actors
        self.args = args

        # build_compile_model 関数用の引数
        model_args = {
            "input_shape": self.args["input_shape"],
            "nb_actions": self.args["nb_actions"],
            "input_sequence": self.args["input_sequence"],
            "enable_dueling_network": self.args["enable_dueling_network"],
            "dueling_network_type": self.args["dueling_network_type"],
            "enable_noisynet": self.args["enable_noisynet"],
            "dense_units_num": self.args["dense_units_num"],
            "lstm_type": self.args["lstm_type"],
            "lstm_units_num": self.args["lstm_units_num"],
            "metrics": self.args["metrics"],
            "create_optimizer_func": create_optimizer_func,
        }
        self._create_process(model_args, args)


    def _create_process(self, model_args, args_org):

        # 各Queueを作成
        experience_q = mp.Queue()
        model_sync_q = [[mp.Queue(), mp.Queue(), mp.Queue()] for _ in range(self.num_actors)]
        self.learner_end_q = [mp.Queue(), mp.Queue()]
        self.actors_end_q = [mp.Queue() for _ in range(self.num_actors)]
        self.learner_logger_q = mp.Queue()
        self.actors_logger_q = mp.Queue()

        # learner ps を作成
        args = (
            model_args,
            self.args,
            experience_q,
            model_sync_q,
            self.learner_end_q,
            self.learner_logger_q,
        )
        if args_org["enable_GPU"]:
            self.learner_ps = mp.Process(target=learner_run_gpu, args=args)
        else:
            self.learner_ps = mp.Process(target=learner_run, args=args)
        
        # actor ps を作成
        self.actors_ps = []
        epsilon = self.args["epsilon"]
        epsilon_alpha = self.args["epsilon_alpha"]
        for i in range(self.num_actors):
            if self.num_actors <= 1:
                epsilon_i = epsilon ** (1 + epsilon_alpha)
            else:
                epsilon_i = epsilon ** (1 + i/(self.num_actors-1)*epsilon_alpha)
            print("Actor{} Epsilon:{}".format(i,epsilon_i))
            self.args["epsilon"] = epsilon_i

            args = (
                i,
                self.actor_func,
                model_args,
                self.args,
                experience_q,
                model_sync_q[i],
                self.actors_logger_q,
                self.actors_end_q[i],
            )
            if args_org["enable_GPU"]:
                self.actors_ps.append(mp.Process(target=actor_run_cpu, args=args))
            else:
                self.actors_ps.append(mp.Process(target=actor_run, args=args))

        # test用 Actor は子 Process では作らないのでselfにする。
        self.model_args = model_args
    
    def __del__(self):
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

    def train(self):
    
        learner_logs = []
        actors_logs = {}
        
        # プロセスを動かす
        try:
            self.learner_ps.start()
            for p in self.actors_ps:
                p.start()

            # 終了を待つ
            while True:
                time.sleep(1)  # polling time

                # 定期的にログを吸出し
                while not self.learner_logger_q.empty():
                    learner_logs.append(self.learner_logger_q.get(timeout=1))
                while not self.actors_logger_q.empty():
                    log = self.actors_logger_q.get(timeout=1)
                    if log["name"] not in actors_logs:
                        actors_logs[log["name"]] = []
                    actors_logs[log["name"]].append(log)
                
                # 終了判定
                f = True
                for q in self.actors_end_q:
                    if q.empty():
                        f = False
                        break
                if f:
                    break
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        
        # 定期的にログを吸出し
        while not self.learner_logger_q.empty():
            learner_logs.append(self.learner_logger_q.get(timeout=1))
        while not self.actors_logger_q.empty():
            log = self.actors_logger_q.get(timeout=1)
            if log["name"] not in actors_logs:
                actors_logs[log["name"]] = []
            actors_logs[log["name"]].append(log)
    
        # learner に終了を投げる
        self.learner_end_q[0].put(1)
        
        # learner から最後の状態を取得
        print("Last Learner weights waiting...")
        weights = self.learner_end_q[1].get(timeout=60)
        
        # test用の Actor を作成
        test_actor = Actor(
            -1,
            self.model_args,
            self.args,
            None,
            None,
        )
        test_actor.model.set_weights(weights)

        # kill
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

        return test_actor, learner_logs, actors_logs


#---------------------------------------------------
# network
#---------------------------------------------------
def rescaling(x, epsilon=0.001):
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x

def rescaling_inverse(x):
    return np.sign(x)*( (x+np.sign(x) ) ** 2 - 1)

def build_compile_model(
    input_shape,         # 入力shape
    input_sequence,      # input_sequence
    nb_actions,          # アクション数
    enable_dueling_network,  # dueling_network を有効にするか
    dueling_network_type,
    enable_noisynet,
    dense_units_num,         # Dense層のユニット数
    lstm_type,              # 使用するLSTMアルゴリズム
    lstm_units_num,         # LSTMのユニット数
    create_optimizer_func,
    metrics,                 # compile に渡す metrics
    ):

    if lstm_type == "lstm_ful":
        # (batch_size, timesteps, width, height)
        c = input_ = Input(batch_shape=(1, 1) + input_shape)
    else:
        # 入力層(input_sequence, width, height)
        c = input_ = Input(shape=(input_sequence,) + input_shape)

    if lstm_type == "":
        c = Flatten()(c)
    if lstm_type == "lstm":
        c = LSTM(lstm_units_num, name="lstm")(c)
    if lstm_type == "lstm_ful":
        c = LSTM(lstm_units_num, stateful=True, name="lstm")(c)

    if enable_dueling_network:
        # value
        v = Dense(dense_units_num, activation="tanh")(c)
        v = Dense(int(dense_units_num/2), activation="tanh")(v)
        v = Dropout(0.1)(v)
        if enable_noisynet:
            v = NoisyDense(1, name="v")(v)
        else:
            v = Dense(1, name="v")(v)

        # advance
        adv = Dense(dense_units_num, activation='tanh')(c)
        adv = Dense(int(dense_units_num/2), activation="tanh")(adv)
        adv = Dropout(0.1)(adv)
        if enable_noisynet:
            adv = NoisyDense(nb_actions, name="adv")(adv)
        else:
            adv = Dense(nb_actions, name="adv")(adv)

        # 連結で結合
        c = Concatenate()([v,adv])
        if dueling_network_type == "ave":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == "max":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == "naive":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_actions,))(c)
        else:
            raise ValueError('dueling_network_type is ["ave","max","naive"]')
    else:
        c = Dense(dense_units_num, activation="tanh")(c)
        if enable_noisynet:
            c = NoisyDense(nb_actions, activation="linear", name="adv")(c)
        else:
            c = Dense(nb_actions, activation="linear", name="adv")(c)
    
    
    model = Model(input_, c)

    # compile
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=create_optimizer_func(), 
        metrics=metrics)
    
    return model

def save_model(model_name, model_group, model):
    model_path = os.path.join("/data/model", model_group)
    save_weights_path = os.path.join(model_path, f"{model_name}.hdf5")
    model.save_weights(save_weights_path, True)
    print(f"save: {save_weights_path}")
    create_graph(model_name, model_group, model)

#---------------------------------------------------
# learner
#---------------------------------------------------
def learner_run_gpu(
    model_args, 
    args, 
    experience_q,
    model_sync_q,
    learner_end_q,
    logger_q,
    ):
    with tf.device("/device:GPU:0"):
        learner_run(
            model_args, 
        args, 
        experience_q,
        model_sync_q,
        learner_end_q,
        logger_q)

def learner_run(
    model_args, 
    args, 
    experience_q,
    model_sync_q,
    learner_end_q,
    logger_q,
    ):
    learner = Learner(
        model_args=model_args, 
        args=args, 
        experience_q=experience_q,
        model_sync_q=model_sync_q,
    )
    try:
        # model load
        if os.path.isfile(args["load_weights_path"]):
            learner.model.load_weights(args["load_weights_path"])
            learner.target_model.load_weights(args["load_weights_path"])

        # logger用
        t0 = time.time()

        # learner はひたすら学習する
        print("Learner Starts!")
        while True:
            learner.train()

            # logger
            if time.time() - t0 > args["logger_interval"]:
                t0 = time.time()
                logger_q.put({
                    "name": "learner",
                    "train_num": learner.train_num,
                })

            # 終了判定
            if not learner_end_q[0].empty():
                break
        else:
            # model load
            if os.path.isfile(args["load_weights_path"]):
                learner.model.load_weights(args["load_weights_path"])
                learner.target_model.load_weights(args["load_weights_path"])

            # logger用
            t0 = time.time()

            # learner はひたすら学習する
            print("Learner Starts!")
            while True:
                learner.train()

                # logger
                if time.time() - t0 > args["logger_interval"]:
                    t0 = time.time()
                    logger_q.put({
                        "name": "learner",
                        "train_num": learner.train_num,
                    })

                # 終了判定
                if not learner_end_q[0].empty():
                    break

    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        print("Learning End. Train Count:{}".format(learner.train_num))
        # model save
        save_model(args["model_name"], args["model_group"], learner.model)

        # 最後の状態を manager に投げる
        print("Last Learner weights sending...")
        learner_end_q[1].put(learner.model.get_weights())


class Learner():
    def __init__(self,
        model_args, 
        args, 
        experience_q,
        model_sync_q
        ):
        self.model_name = args["model_name"]
        self.model_group = args["model_group"]
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q

        self.memory_warmup_size = args["remote_memory_warmup_size"]

        self.per_alpha = args["per_alpha"]
        per_beta_initial = args["per_beta_initial"]
        per_beta_steps = args["per_beta_steps"]
        per_enable_is = args["per_enable_is"]
        memory_capacity = args["remote_memory_capacity"]
        memory_type = args["remote_memory_type"]
        if memory_type == "replay":
            self.memory = ReplayMemory(memory_capacity)
        elif memory_type == "per_greedy":
            self.memory = PERGreedyMemory(memory_capacity)
        elif memory_type == "per_proportional":
            self.memory = PERProportionalMemory(memory_capacity, per_beta_initial, per_beta_steps, per_enable_is)
        elif memory_type == "per_rankbase":
            self.memory = PERRankBaseMemory(memory_capacity, self.per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        else:
            raise ValueError('memory_type is ["replay","per_proportional","per_rankbase"]')

        self.gamma = args["gamma"]
        self.batch_size = args["batch_size"]
        self.enable_double_dqn = args["enable_double_dqn"]
        self.target_model_update = args["target_model_update"]
        self.multireward_steps = args["multireward_steps"]
        self.input_sequence = args["input_sequence"]
        self.lstm_type = args["lstm_type"]
        self.enable_rescaling_priority = args["enable_rescaling_priority"]
        self.enable_rescaling_train = args["enable_rescaling_train"]
        self.rescaling_epsilon = args["rescaling_epsilon"]
        self.burnin_length = args["burnin_length"]
        self.priority_exponent = args["priority_exponent"]

        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.memory_warmup_size > self.batch_size, "Warmup steps is few.(Larger than batch size)"

        # local
        self.train_num = 0

        # model create
        self.model = build_compile_model(**model_args)
        self.model.summary()
        self.target_model = build_compile_model(**model_args)

        # lstm ful では lstmレイヤーを使う
        if self.lstm_type == "lstm_ful":
            self.lstm = self.model.get_layer("lstm")
            self.target_lstm = self.target_model.get_layer("lstm")


    def train(self):
        
        # Actor から要求があれば weights を渡す
        for q in self.model_sync_q:
            if not q[0].empty():
                # 空にする(念のため)
                while not q[0].empty():
                    q[0].get(timeout=1)
                
                # 送る
                q[1].put(self.model.get_weights())
        
        # experience があれば RemoteMemory に追加
        while not self.experience_q.empty():
            exps = self.experience_q.get(timeout=1)
            for exp in exps:
                if self.lstm_type == "lstm_ful":
                    self.memory.add(exp, exp[4])
                else:
                    self.memory.add(exp, exp[4])

        # RemoteMemory が一定数貯まるまで学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            time.sleep(1)  # なんとなく
            return
        
        (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.train_num)
        # 学習(長いので関数化)
        if self.lstm_type == "lstm_ful":
            self.train_model_ful(indexes, batchs, weights)
        else:
            self.train_model(indexes, batchs, weights)
        self.train_num += 1

        # target networkの更新
        if self.train_num % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            save_model(self.model_name, self.model_group, self.target_model)

    
    # ノーマルの学習
    def train_model(self, indexes, batchs, weights):
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for batch in batchs:
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])

        # 更新用に現在のQネットワークを出力(Q network)
        outputs = self.model.predict(np.asarray(state0_batch), self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_model_qvals_batch = self.model.predict(np.asarray(state1_batch), self.batch_size)
            state1_target_qvals_batch = self.target_model.predict(np.asarray(state1_batch), self.batch_size)
        else:
            # 次の状態のQ値を取得(target_network)
            target_qvals = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

        for i in range(self.batch_size):
            if self.enable_double_dqn:
                action = np.argmax(state1_model_qvals_batch[i])  # modelからアクションを出す
                maxq = state1_target_qvals_batch[i][action]  # Q値はtarget_modelを使って出す
            else:
                maxq = np.max(target_qvals[i])

            # priority計算
            if self.enable_rescaling_priority:
                tmp = rescaling_inverse(maxq)
            else:
                tmp = maxq
            tmp = reward_batch[i] + (self.gamma ** self.multireward_steps) * tmp
            tmp *= weights[i]
            if self.enable_rescaling_priority:
                tmp = rescaling(tmp, self.rescaling_epsilon)
            priority = abs(tmp - outputs[i][action_batch[i]]) ** self.per_alpha

            # Q値の更新
            if self.enable_rescaling_train:
                maxq = rescaling_inverse(maxq)
            td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
            td_error *= weights[i]
            if self.enable_rescaling_train:
                td_error = rescaling(td_error, self.rescaling_epsilon)
            outputs[i][action_batch[i]] = td_error

            # priorityを更新を更新
            self.memory.update(indexes[i], batchs[i], priority)

        # 学習
        self.model.train_on_batch(np.asarray(state0_batch), np.asarray(outputs))
        #self.model.fit(np.asarray(state0_batch), np.asarray(outputs), batch_size=self.batch_size, epochs=1, verbose=0)


    # ステートフルLSTMの学習
    def train_model_ful(self, indexes, batchs, weights):

        # 各経験毎に処理を実施
        for batch_i, batch in enumerate(batchs):
            states = batch[0]
            action = batch[1]
            reward = batch[2]
            hidden_state = batch[3]
            prioritys = []

            # burn-in
            self.lstm.reset_states(hidden_state)
            for i in range(self.burnin_length):
                self.model.predict(np.asarray([[states[i]]]), 1)
            # burn-in 後の結果を保存
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
        
            # 以降は1sequenceずつ更新させる
            for i in range(self.input_sequence):
                state0 = [states[self.burnin_length + i]]
                state1 = [states[self.burnin_length + i + self.multireward_steps]]

                # 現在のQネットワークを出力
                self.lstm.reset_states(hidden_state)
                output = self.model.predict(np.asarray([state0]), 1)[0]

                # TargetネットワークとQネットワークの値を出力
                if self.enable_double_dqn:
                    self.lstm.reset_states(hidden_state)
                    self.target_lstm.reset_states(hidden_state)
                    state1_model_qvals = self.model.predict(np.asarray([state1]), 1)[0]
                    state1_target_qvals = self.target_model.predict(np.asarray([state1]), 1)[0]

                    action_q = np.argmax(state1_model_qvals)
                    maxq = state1_target_qvals[action_q]

                else:
                    self.target_lstm.reset_states(hidden_state)
                    target_qvals = self.target_model.predict(np.asarray([state1], 1))[0]
                    maxq = np.max(target_qvals)

                # priority計算
                if self.enable_rescaling_priority:
                    tmp = rescaling_inverse(maxq)
                else:
                    tmp = maxq
                tmp = reward[i] + (self.gamma ** self.multireward_steps) * tmp
                tmp *= weights[batch_i]
                if self.enable_rescaling_priority:
                    tmp = rescaling(tmp)
                priority = abs(tmp - output[action[i]]) ** self.per_alpha
                prioritys.append(priority)
                
                # Q値 update用
                if self.enable_rescaling_train:
                    maxq = rescaling_inverse(maxq)
                td_error = reward[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[batch_i]
                if self.enable_rescaling_train:
                    td_error = rescaling(td_error, self.rescaling_epsilon)
                output[action[i]] = td_error

                # 学習
                self.lstm.reset_states(hidden_state)
                self.model.fit(
                    np.asarray([state0]), 
                    np.asarray([output]), 
                    batch_size=1, 
                    epochs=1, 
                    verbose=0, 
                    shuffle=False
                )

                # 次の学習用に hidden state を保存
                hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]

            # 今回使用したsamplingのpriorityを更新
            priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)
            self.memory.update(indexes[batch_i], batch, priority)
        


#---------------------------------------------------
# actor
#---------------------------------------------------

def actor_run_cpu(
    actor_index,
    actor_func, 
    model_args,
    args,
    experience_q, 
    model_sync_q, 
    logger_q,
    actors_end_q,
    ):
    with tf.device("/device:CPU:0"):
        actor_run(
            actor_index,
            actor_func, 
            model_args,
            args,
            experience_q, 
            model_sync_q, 
            logger_q,
            actors_end_q)

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
    print("Actor{} Starts!".format(actor_index))
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
        print("Actor{} End!".format(actor_index))
        actors_end_q.put(1)


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
        self.input_sequence = args["input_sequence"]
        self.actor_model_sync_interval = args["actor_model_sync_interval"]
        self.gamma = args["gamma"]
        self.epsilon = args["epsilon"]
        self.multireward_steps = args["multireward_steps"]
        self.action_interval = args["action_interval"]
        self.burnin_length = args["burnin_length"]
        self.lstm_type = args["lstm_type"]
        self.enable_dueling_network = args["enable_dueling_network"]
        self.enable_noisynet = args["enable_noisynet"]
        self.enable_rescaling_priority = args["enable_rescaling_priority"]
        self.rescaling_epsilon = args["rescaling_epsilon"]
        self.priority_exponent = args["priority_exponent"]

        self.per_alpha = args["per_alpha"]

        # local memory
        self.local_memory = deque()
        self.local_memory_update_size = args["local_memory_update_size"]

        self.learner_train_num = 0
        
        # model
        self.model = build_compile_model(**model_args)
        # lstm ful では lstmレイヤーを使う
        if self.lstm_type == "lstm_ful":
            self.lstm = self.model.get_layer("lstm")

        self.compiled = True

    def reset_states(self):
        self.repeated_action = 0

        self.recent_action = [ 0 for _ in range(self.input_sequence)]
        self.recent_reward = [ 0 for _ in range(self.input_sequence + self.multireward_steps - 1)]
        obs_length = self.burnin_length + self.input_sequence + self.multireward_steps
        self.recent_observations = [np.zeros(self.input_shape) for _ in range(obs_length)]

        if self.lstm_type == "lstm_ful":
            self.model.reset_states()
            self.recent_hidden_state = [
                [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
                for _ in range(self.burnin_length + self.input_sequence)
            ]

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def load_weights(self, filepath):
        print("WARNING: Not Loaded. Please use 'load_weights_path' param.")

    def save_weights(self, filepath, overwrite=False):
        print("WARNING: Not Saved. Please use 'save_weights_path' param.")

    def forward(self, observation):
        self.recent_observations.append(observation)  # 最後に追加
        self.recent_observations.pop(0)  # 先頭を削除

        if self.training:

            #--- 経験を送る
            if self.lstm_type == "lstm_ful":
                
                # priorityを計算
                prioritys = []
                rewards = []
                for i in range(self.input_sequence):
                    state0 = [self.recent_observations[self.burnin_length + i]]
                    state1 = [self.recent_observations[self.burnin_length + i + self.multireward_steps]]
                    hidden_state = self.recent_hidden_state[i]
                    action = self.recent_action[i]

                    # Multi-Step learning
                    reward = 0
                    for j in range(self.multireward_steps):
                        reward += self.recent_reward[i+j] * (self.gamma ** j)
                    rewards.append(reward)

                    # 現在のQネットワークを出力
                    self.lstm.reset_states(hidden_state)
                    state0_qvals = self.model.predict(np.asarray([state0]), 1)[0]
                    self.lstm.reset_states(hidden_state)
                    state1_qvals = self.model.predict(np.asarray([state1]), 1)[0]

                    maxq = np.max(state1_qvals)

                    # priority計算
                    if self.enable_rescaling_priority:
                        td_error = rescaling_inverse(maxq)
                    td_error = reward + (self.gamma ** self.multireward_steps) * maxq
                    if self.enable_rescaling_priority:
                        td_error = rescaling(td_error, self.rescaling_epsilon)
                    priority = abs(td_error - state0_qvals[action]) ** self.per_alpha
                    prioritys.append(priority)
                
                # 今回使用したsamplingのpriorityを更新
                priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)
                
                # local memoryに追加
                self.local_memory.append((
                    self.recent_observations[:],
                    self.recent_action[:],
                    rewards, 
                    self.recent_hidden_state[0],
                    priority,
                ))

            else:
                # Multi-Step learning
                reward = 0
                for i, r in enumerate(reversed(self.recent_reward)):
                    reward += r * (self.gamma ** i)

                state0 = self.recent_observations[self.burnin_length:self.burnin_length+self.input_sequence]
                state1 = self.recent_observations[-self.input_sequence:]
                action = self.recent_action[-1]

                # priority のために TD-error をだす。
                state0_qvals = self.model.predict(np.asarray([state0]), 1)[0]
                state1_qvals = self.model.predict(np.asarray([state1]), 1)[0]
                maxq = np.max(state1_qvals)

                # priority計算
                if self.enable_rescaling_priority:
                    td_error = rescaling(maxq) ** -1
                td_error = reward + (self.gamma ** self.multireward_steps) * maxq
                if self.enable_rescaling_priority:
                    td_error = rescaling(td_error, self.rescaling_epsilon)
                priority = abs(td_error - state0_qvals[action]) ** self.per_alpha
            
                # local memoryに追加
                self.local_memory.append((
                    state0, 
                    action, 
                    reward, 
                    state1, 
                    priority,
                ))
        
        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:

            if self.lstm_type == "lstm_ful":
                # 状態を復元
                self.lstm.reset_states(self.recent_hidden_state[-1])
            
            # 行動を決定
            action = self.select_action()

            if self.lstm_type == "lstm_ful":
                # 状態を保存
                self.recent_hidden_state.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])
                self.recent_hidden_state.pop(0)
            
            # リピート用
            self.repeated_action = action

        # backword用に保存
        self.recent_action.append(action)  # 最後に追加
        self.recent_action.pop(0)  # 先頭を削除
        
        return action

    # 長いので関数に
    def select_action(self):
        # noisy netが有効の場合はそちらで探索する
        if self.training and not self.enable_noisynet:
            
            # ϵ-greedy法
            if self.epsilon > np.random.uniform(0, 1):
                # ランダム
                action = np.random.randint(0, self.nb_actions)
            else:
                action = self._get_qmax_action()
        else:
            action = self._get_qmax_action()

        return action

    # 2箇所あるので関数に、現状の最大Q値のアクションを返す
    def _get_qmax_action(self):

        if self.lstm_type == "lstm_ful":
            # 最後の状態のみ
            state1 = [self.recent_observations[-1]]
            q_values = self.model.predict(np.asarray([state1]), batch_size=1)[0]
        else:
            # sequence分の入力
            state1 = self.recent_observations[-self.input_sequence:]
            q_values = self.model.predict(np.asarray([state1]), batch_size=1)[0]

        return np.argmax(q_values)


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

#--------------------------------------------------------

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.buffer = []

    def add(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, priority):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)


#--------------------------------------------------------

import heapq
class _head_wrapper():
    def __init__(self, data):
        self.d = data
    def __eq__(self, other):
        return True

class PERGreedyMemory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience, priority):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        # priority は最初は最大を選択
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))

    def update(self, idx, experience, priority):
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [heapq.heappop(self.buffer)[1].d for _ in range(batch_size)]

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

#---------------------------------------------------

#copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERProportionalMemory():
    def __init__(self, capacity, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, priority):
        self.tree.add(priority, experience)

    def update(self, index, experience, priority):
        self.tree.update(index, priority)

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
    
        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section*i + random.random()*section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights)

    def __len__(self):
        return self.tree.write

#------------------------------------

import bisect
class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0
    def __lt__(self, o):  # a<b
        return self.priority > o.priority

class PERRankBaseMemory():
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, priority):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, priority):
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)


    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps

        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha 
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section*i + random.random()*section)
        
        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                priority = o.p
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

#----------------------------------------

# from : https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# from : https://github.com/keiohta/tf2rl/blob/atari/tf2rl/networks/noisy_dense.py
class NoisyDense(Layer):

    def __init__(self, units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel'
                                      )


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias = None
            self.epsilon_bias = None

        self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias = K.zeros(shape=(self.units,))

        self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        #perturbation = self.sigma_kernel * self.epsilon_kernel
        #perturbed_kernel = self.kernel + perturbation
        perturbed_kernel = self.sigma_kernel * K.random_uniform(shape=self.kernel_shape)

        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            #bias_perturbation = self.sigma_bias * self.epsilon_bias
            #perturbed_bias = self.bias + bias_perturbation
            perturbed_bias = self.bias + self.sigma_bias * K.random_uniform(shape=self.bias_shape)
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))


#---------------------------------------------------

def create_optimizer():
    return Adam(lr=0.00025)

def train_r2d2(csvpath, model_name, model_group, nb_steps, num_actors=6, mode="stg"):
    nb_actions = 3
    csvfiles = glob(csvpath)
    env = Market(csvfiles, nb_actions, debug=False, mode=mode)
    def actor_func(index, actor):
        if index == 0:
            verbose = 1
        else:
            verbose = 0
        actor.fit(env, nb_steps=nb_steps, visualize=False, verbose=verbose)

    # 引数
    args = {
        # model関係
        "input_shape": env.observation_space.shape, 
        "nb_actions": 3, 
        "input_sequence": 1,     # 入力フレーム数
        "dense_units_num": 32,  # Dense層のユニット数
        "metrics": [],           # optimizer用
        "enable_dueling_network": True,  # dueling_network有効フラグ
        "dueling_network_type": "ave",   # dueling_networkのアルゴリズム
        "enable_noisynet": False,        # NoisyNet有効フラグ
        "lstm_type": "lstm_ful",             # LSTMのアルゴリズム
        "lstm_units_num": 64,   # LSTM層のユニット数
        
        # learner 関係
        "remote_memory_capacity": 1000000,    # 確保するメモリーサイズ
        "remote_memory_warmup_size": 10000,    # 初期のメモリー確保用step数(学習しない)
        "remote_memory_type": "per_proportional", # メモリの種類
        "per_alpha": 0.6,            # PERの確率反映率
        "per_beta_initial": 0.4,     # IS反映率の初期値
        "per_beta_steps": 1000000,   # IS反映率の上昇step数
        "per_enable_is": True,       # ISを有効にするかどうか
        "batch_size": 64,            # batch_size
        "target_model_update": 32, #  target networkのupdate間隔
        "enable_double_dqn": True,   # DDQN有効フラグ
        "enable_rescaling_priority": True,   # rescalingを有効にするか(priotrity)
        "enable_rescaling_train": True,      # rescalingを有効にするか(train)
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "burnin_length": 36,        # burn-in期間
        "priority_exponent": 0.9,    # priority優先度

        # actor 関係
        "local_memory_update_size": 50,    # LocalMemoryからRemoteMemoryへ投げるサイズ
        "actor_model_sync_interval": 500,  # learner から model を同期する間隔
        "gamma": 0.99,      # Q学習の割引率
        "epsilon": 0.1,        # ϵ-greedy法
        "epsilon_alpha": 2,    # ϵ-greedy法
        "multireward_steps": 512, # multistep reward
        "action_interval": 1,   # アクションを実行する間隔
        
        # その他
        "load_weights_path": f"/data/model/{model_group}/{model_name}.hdf5",  # 保存ファイル名
        "save_weights_path": f"/data/model/{model_group}/{model_name}.hdf5",  # 読み込みファイル名
        "save_overwrite": True,   # 上書き保存するか
        "logger_interval": 10,    # ログ取得間隔(秒)
        "enable_GPU": False,      # GPUを使うか,
        "model_name": model_name,
        "model_group": model_group
    }


    manager = R2D2Manager(
        actor_func=actor_func, 
        num_actors=num_actors,    # actor数
        args=args, 
        create_optimizer_func=create_optimizer,
    )
    
    agent, learner_logs, actors_logs = manager.train()
