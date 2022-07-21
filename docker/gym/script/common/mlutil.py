import pandas as pd
import numpy as np

def create_dataset(csvfile, test_size=0.5):
    all_test_df = None
    all_train_df = None
    # データフレームの読み込み
    df = pd.read_csv(csvfile)

    # Dataサイズ
    data_len = len(df)

    # 教師データとテストデータを分割する
    train_data_rate = 1 - test_size
    train_df = df[:int(data_len*train_data_rate)]
    test_df = df[int(data_len*train_data_rate):]
    return (train_df, test_df)
