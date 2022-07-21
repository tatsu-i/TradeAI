#---------------------------------------------------
# manager
#---------------------------------------------------
import os
import time
import traceback
import multiprocessing as mp
from .learner import learner_run
from .actor import actor_run, Actor

class ApeXManager():
    def __init__(self,
        actor_func,
        num_actors,
        args,
        create_optimizer_func,
        ):

        self.actor_func = actor_func
        self.num_actors = num_actors
        self.args = args

        # build_compile_model 関数用の引数
        model_args = {
            "input_shape": self.args["input_shape"],
            "enable_image_layer": self.args["enable_image_layer"],
            "nb_actions": self.args["nb_actions"],
            "window_length": self.args["window_length"],
            "enable_dueling_network": self.args["enable_dueling_network"],
            "dense_units_num": self.args["dense_units_num"],
            "metrics": self.args["metrics"],
            "create_optimizer_func": create_optimizer_func,
        }
        self._create_process(model_args)


    def _create_process(self, model_args):

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
