import os
import time
import traceback
import numpy as np
from .memory import *
from .network import build_compile_model
from common.api import save_model
#---------------------------------------------------
# learner
#---------------------------------------------------

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
            print(f"load weights {args['load_weights_path']}")
            learner.model.load_weights(args["load_weights_path"])
            learner.target_model.load_weights(args["load_weights_path"])

        # logger用
        t0 = time.time()

        # learner はひたすら学習する
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
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q
        self.model_name = args["model_name"]
        self.model_group = args["model_group"]

        self.memory_warmup_size = args["remote_memory_warmup_size"]

        per_alpha = args["per_alpha"]
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
            self.memory = PERProportionalMemory(memory_capacity, per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        elif memory_type == "per_rankbase":
            self.memory = PERRankBaseMemory(memory_capacity, per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        else:
            raise ValueError('memory_type is ["replay","per_proportional","per_rankbase"]')

        self.gamma = args["gamma"]
        self.batch_size = args["batch_size"]
        self.enable_double_dqn = args["enable_double_dqn"]
        self.target_model_update = args["target_model_update"]
        self.multireward_steps = args["multireward_steps"]

        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.memory_warmup_size > self.batch_size, "Warmup steps is few.(Larger than batch size)"

        # local
        self.train_num = 0

        # model create
        self.model = build_compile_model(**model_args)
        self.target_model = build_compile_model(**model_args)


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
                self.memory.add(exp, exp[4])

        # RemoteMemory が一定数貯まるまで学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            time.sleep(1)  # なんとなく
            return

        (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.train_num)
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

            for i in range(self.batch_size):
                action = np.argmax(state1_model_qvals_batch[i])  # modelからアクションを出す
                maxq = state1_target_qvals_batch[i][action]  # Q値はtarget_modelを使って出す

                td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[i]
                td_error_diff = outputs[i][action_batch[i]] - td_error  # TD誤差を取得
                outputs[i][action_batch[i]] = td_error   # 更新

                # TD誤差を更新
                self.memory.update(indexes[i], batchs[i], td_error_diff)

        else:
            # 次の状態のQ値を取得(target_network)
            target_qvals = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

            # Q学習、Q(St,At)=Q(St,At)+α(r+γmax(St+1,At+1)-Q(St,At))
            for i in range(self.batch_size):
                maxq = np.max(target_qvals[i])
                td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[i]
                td_error_diff = outputs[i][action_batch[i]] - td_error  # TD誤差を取得
                outputs[i][action_batch[i]] = td_error

                self.memory.update(batchs[i], td_error_diff)

        # 学習
        #self.model.train_on_batch(np.asarray(state0_batch), np.asarray(outputs))
        self.model.fit(np.asarray(state0_batch), np.asarray(outputs), batch_size=self.batch_size, epochs=1, verbose=0)
        self.train_num += 1

        # target networkの更新
        if self.train_num % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            save_model(self.model_name, self.model_group, self.target_model)

