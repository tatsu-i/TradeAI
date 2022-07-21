import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない
from glob import glob
from keras.optimizers import Adam
from .manager import ApeXManager
from .learner import Learner
from common.gymenv import Market

#base_path = "/content/drive/My Drive"
base_path = "/data"
# feature version
version="1.1"

def create_optimizer():
    return Adam()

def train_apex(csvpath, model_name, model_group, nb_steps, num_actors=6, mode="stg"):
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
        "enable_image_layer": False, 
        "nb_actions": nb_actions, 
        "window_length": 1,     # 入力フレーム数
        "dense_units_num": 32,  # Dense層のユニット数
        "metrics": [],          # optimizer用
        "enable_dueling_network": True,  # dueling_network有効フラグ

        # learner 関係
        "remote_memory_capacity": 600000,    # 確保するメモリーサイズ
        "remote_memory_warmup_size": 1000,    # 初期のメモリー確保用step数(学習しない)
        "remote_memory_type": "per_proportional", # メモリの種類
        "per_alpha": 0.8,        # PERの確率反映率
        "per_beta_initial": 0.0,     # IS反映率の初期値
        "per_beta_steps": 5000,   # IS反映率の上昇step数
        "per_enable_is": True,      # ISを有効にするかどうか
        "batch_size": 512,            # batch_size
        "target_model_update": 1000, #  target networkのupdate間隔
        "enable_double_dqn": True,   # DDQN有効フラグ

        # actor 関係
        "local_memory_update_size": 50,    # LocalMemoryからRemoteMemoryへ投げるサイズ
        "actor_model_sync_interval": 1000,  # learner から model を同期する間隔
        "gamma": 0.6,      # Q学習の割引率
        "epsilon": 0.1,        # ϵ-greedy法
        "epsilon_alpha": 9,    # ϵ-greedy法
        "multireward_steps": 12, # multistep reward
        "action_interval": 1,   # アクションを実行する間隔

        # その他
        "load_weights_path": f"/data/model/{model_group}/{model_name}.hdf5",  # 保存ファイル名
        "save_overwrite": True,   # 上書き保存するか
        "logger_interval": 100,    # ログ取得間隔(秒)
        "model_name": model_name,
        "model_group": model_group,
    }

    manager = ApeXManager(
        actor_func=actor_func, 
        num_actors=num_actors, 
        args=args, 
        create_optimizer_func=create_optimizer,
    )
    test_actor, learner_logs, actors_logs = manager.train()
