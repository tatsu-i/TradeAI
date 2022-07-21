import os
import pickle
import tensorflow as tf
from keras import backend as K

# RedisAIにインポートするためにfrozen graph形式に変換
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# KerasModelをProtocol Buffer形式に変換
def create_graph(model_name, model_group, model):
    try:
        base_path = os.environ.get("MODEL_PATH", "/data/model")
        model_path = os.path.join(base_path, model_group)
        frozen_graph = freeze_session(
            K.get_session(),
            output_names=[out.op.name for out in model.outputs]
        )
        inputs = [ inp.op.name for inp in model.inputs]
        outputs = [out.op.name for out in model.outputs]
        args = {"inputs": inputs, "outputs": outputs}
        pkl_path = os.path.join(model_path, f"{model_name}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(args, f)
        graph_file = tf.train.write_graph(
            frozen_graph,
            model_path, 
            f"{model_name}.pb",
            as_text=False
        )
        print(f"dump graph {graph_file}")
    except Exception as e:
        print(f"Error {model_name} {e}")
        return 1
    return 0

def save_model(model_name, model_group, model):
    base_path = os.environ.get("MODEL_PATH", "/data/model")
    model_path = os.path.join(base_path, model_group)
    save_weights_path = os.path.join(model_path, f"{model_name}.hdf5")
    model.save_weights(save_weights_path, True)
    print(f"save: {save_weights_path}")
    create_graph(model_name, model_group, model)
