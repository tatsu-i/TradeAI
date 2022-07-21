import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import click
import pickle
import ml2rt
import tensorflow as tf
from redisai import Client, Device, Backend

base_path = os.environ.get("MODEL_PATH", "/data/model")

# Redisにモデルを読み込む
def deploy_model(model_name, model_group, redis_host):
    try:
        client = Client(redis_host, decode_responses=False)
        model_path = os.path.join(base_path, model_group)
        graph_file = os.path.join(model_path, f"{model_name}.pb")
        pkl_path = os.path.join(model_path, f"{model_name}.pkl")
        with open(pkl_path, 'rb') as f:
            args = pickle.load(f)
        ml2rt_model = ml2rt.load_model(graph_file)
        client.delete(model_name)
        client.modelset(
            model_name, 
            Backend.tf, 
            Device.cpu,
            inputs=args["inputs"], 
            outputs=args["outputs"],
            data=ml2rt_model
        )
        print(f"Load model {graph_file}")
    except Exception as e:
        print(f"deploy model error {model_name} {e}")
        return 1
    return 0

@click.command()
@click.option('--name', '-n', default='default')
@click.option('--group', '-g', default='best')
@click.option('--redis_host', '-r', default='redisai')
def main(name, group, redis_host):
    deploy_model(name, group, redis_host)

if __name__ == '__main__':
    main()
