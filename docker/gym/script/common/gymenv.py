import gym.spaces
import numpy as np
import random
from keras.utils.np_utils import to_categorical

import ephem
from datetime import datetime
from pytz import timezone
from .mlutil import create_dataset

# 日付から特徴を抽出する
def lunar_age(timestamp):
    d = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    # 指定時刻をUTCに変換
    jst = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    local_tz = timezone('Asia/Tokyo')
    jst = local_tz.localize(jst)
    utc = jst.astimezone(timezone('UTC'))
    # 前回新月時刻を求め結果をJSTに変換
    previous_new_moon_time = ephem.previous_new_moon(utc).datetime()
    previous_new_moon_time = timezone('UTC').localize(previous_new_moon_time)
    previous_new_moon_time_jst = previous_new_moon_time.astimezone(timezone('Asia/Tokyo'))
    # 前回新月時刻から指定時刻までの経過時間から月齢を求める（小数点２位以下は四捨五入）
    lunar_date = jst - previous_new_moon_time_jst
    lunar_date = round(lunar_date.days + lunar_date.seconds/(60*60*24.), 1)
    return lunar_date

def destiny(timestamp):
    destiny_table = {
        0:  "種子",
        1:  "緑生",
        2:  "立花",
        3:  "健弱",
        4:  "達成",
        5:  "乱気",
        6:  "再開",
        7:  "財政",
        8:  "安定",
        9:  "陰影",
        10: "停止",
        11: "減退",
    }
    local_tz = timezone('Asia/Tokyo')

    base_timestamp = "1970/01/01 00:00:00"
    d = datetime.strptime(base_timestamp, "%Y/%m/%d %H:%M:%S")
    jst = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    jst_base = local_tz.localize(jst)

    d = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    jst = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    jst_target = local_tz.localize(jst)

    dt_offset = jst_target - jst_base
    index = (dt_offset.days + 10) % 12
    label = destiny_table[index]
    return index

def dateparse(timestamp):
    local_tz = timezone('Asia/Tokyo')
    d = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    jst = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    return local_tz.localize(jst)

# 学習環境の定義
class Market(gym.core.Env):
    def __init__(
        self,
        csvfiles,
        n_action=4,
        capital = 1000000,
        debug=False,
        mode="stg"
    ):
        self.info = {}
        self.csvfiles = csvfiles
        self.action_space = gym.spaces.Discrete(n_action)  # 行動空間
        self.debug = debug
        self.get_observation_space()
        self.game_over_profit = -300000
        self.init_capital = capital
        self.mode = mode
        self.reset()
        
    def reset(self):
        csvfile = random.choice(self.csvfiles)
        train_df, test_df = create_dataset(
            csvfile,
            test_size=0.5
        )
        print(f"select {csvfile}")
        self.df = train_df if self.mode == "stg" else test_df
        self.capital = self.init_capital
        self.profit = 0
        self.time = 0
        self.h = 0
        self.l = 0
        self.c = 0
        self.StockUnit = 0
        self.OrderUnit = 0
        self.OrderPosition = 0
        self.OrderPrice = 0
        self.LECount = 0
        self.SECount = 0
        self.OrderMiss = 0
        self.MarketPosition = 0
        self.EntryPrice = 0
        self.EntryTime = 0
        self.ExitCount = 0
        self.FlatCount = 0
        self.MaxHoldCount = 0
        self.BarsSinceEntry = 0
        self.flat_status = 0
        self.game_over_profit += self.profit
        self.game_over_profit = -300000 if self.game_over_profit >= 0 else self.game_over_profit
        self.ProcessOrder()
        return self.get_observation()
    
    def get_observation_space(self): 
        tech_feature_length=69
        tech_opt_feature_length=5
        feature_length=0
        self.observable_columns = [f"Value{i+1}" for i in range(tech_feature_length-tech_opt_feature_length)]
        shape_high = []
        shape_low  = []
        # Value1 〜 Value69
        feature_length += tech_feature_length
        for i in range(tech_feature_length):
            shape_high.append(1)
            shape_low.append(-1)
        # 月
        #feature_length += 12
        #for i in range(12):
        #    shape_high.append(1)
        #    shape_low.append(0)
        # 時間
        feature_length += 24
        for i in range(24):
            shape_high.append(1)
            shape_low.append(0)
        # 六星占術
        feature_length += 12
        for i in range(12):
            shape_high.append(1)
            shape_low.append(0)
        # 月齢
        feature_length += 30
        for i in range(30):
            shape_high.append(1)
            shape_low.append(0)
        self.shape = (feature_length,)
        self.observation_space = gym.spaces.Box(low=np.array(shape_low), high=np.array(shape_high))

    def get_observation(self):
        timestamp = self.df.iloc[self.time]['timestamp']
        # Value1 〜 ValueN
        observation = self.df[self.observable_columns].iloc[self.time].values
        # ValueN 〜 Value69
        observation = np.append(observation, self.FeatureTargetPositionProfit(10000,30000))
        observation = np.append(observation, self.FeatureBarsSinceEntry(62))
        observation = np.append(observation, self.FeatureBarsSinceEntry(36))
        observation = np.append(observation, self.FeatureOpenPositionProfit())
        observation = np.append(observation, self.MarketPosition)
        # 月
        #for month in list(to_categorical(dateparse(timestamp).month-1, num_classes=12)):
        #    observation = np.append(observation, month)
        # 時間
        for hour in list(to_categorical(dateparse(timestamp).hour, num_classes=24)):
            observation = np.append(observation, hour)
        # 六星占術
        for index in list(to_categorical(destiny(timestamp), num_classes=12)):
            observation = np.append(observation, index)
        # 月齢
        for age in list(to_categorical(lunar_age(timestamp), num_classes=30)):
            observation = np.append(observation, age)
        return observation
    
    def FeatureOpenPositionProfit(self):
        profit = 0
        if self.MarketPosition == 1:
            profit = (self.c - self.EntryPrice) * self.StockUnit - 400
        if self.MarketPosition == -1:
            profit = (self.EntryPrice - self.c) * self.StockUnit - 400
        if profit > 0:
            return 1
        if profit < 0:
            return -1
        return 0
    
    def FeatureTargetPositionProfit(self, ProfitTarget, StopLoss):
        profit = 0
        if self.MarketPosition == 1:
            profit = (self.c - self.EntryPrice) * self.StockUnit - 400
        if self.MarketPosition == -1:
            profit = (self.EntryPrice - self.c) * self.StockUnit - 400
        if profit > ProfitTarget:
            return 1
        if profit < -StopLoss:
            return -1
        return 0
    
    def FeatureBarsSinceEntry(self, numBars, minBars=3):
        if self.MarketPosition == 0:
            return 0
        if minBars < self.BarsSinceEntry <= numBars:
            return 1
        if self.BarsSinceEntry > numBars:
            return -1
        return 0
    
    def render(self, mode):
        pass

    def close(self):
        pass

    def log(self, msg):
        if self.debug:
            print(msg)

    def LongEntry(self, price):
        self.OrderPrice = price
        self.OrderPosition = 1
        self.OrderUnit = self.GetStockNum(self.c)

    def ShortEntry(self, price):
        self.OrderPrice = price
        self.OrderPosition = -1
        self.OrderUnit = self.GetStockNum(self.c)

    def LongExit(self, price):
        profit = 0
        #if random.randint(1,10) <= 6:
        #    return profit
        if self.MarketPosition == 1:
            if price <= self.h:
                profit = (price - self.EntryPrice) * self.StockUnit - 400
                self.MarketPosition = 0
                self.StockUnit = 0
                self.ExitCount += 1
                self.log(
                    f"{self.time} {self.df.iloc[self.time]['timestamp']} AI LX {self.profit}"
                )
        return profit

    def ShortExit(self, price):
        profit = 0
        #if random.randint(1,10) <= 6:
        #    return profit
        if self.MarketPosition == -1:
            if self.l <= price:
                profit = (self.EntryPrice - price) * self.StockUnit - 400
                self.MarketPosition = 0
                self.StockUnit = 0
                self.ExitCount += 1
                self.log(
                    f"{self.time} {self.df.iloc[self.time]['timestamp']} AI SX {self.profit}"
                )
        return profit


    # 時間を進めずに未来を見る
    def clairvoyants(self):
        profit = 0
        reward = 0
        can_entry = False
        if self.MarketPosition != 0:
            return reward
        # Entry
        try:
            h = self.df.iloc[self.time+1]["high"]
            l = self.df.iloc[self.time+1]["low"]
            c = self.df.iloc[self.time+1]["close"]
            if self.MarketPosition == 0 and self.OrderPosition == 1:
                if self.l <= self.OrderPrice:
                    can_entry = True
            if self.MarketPosition == 0 and self.OrderPosition == -1:
                if self.OrderPrice <= self.h:
                    can_entry = True
        except:
            pass
        
        if not can_entry:
            return 0
        
        for i in range(2,37):
            try:
                h = self.df.iloc[self.time+i]["high"]
                l = self.df.iloc[self.time+i]["low"]
                c = self.df.iloc[self.time+i]["close"]
                if self.OrderPosition == 1:
                    if self.c <= h:
                        profit = h - self.c
                if self.OrderPosition == -1:
                    if l <= self.c:
                        profit = self.c - l
                if reward < profit:
                    reward = profit
            except:
                pass
        return reward * self.OrderUnit - 400
    
    def GetStockNum(self, price, unit=100):
        return int(self.capital / price) - int((self.capital / price)%unit)
    
    def ProcessOrder(self):
        self.h = self.df.iloc[self.time]["high"]
        self.l = self.df.iloc[self.time]["low"]
        self.c = self.df.iloc[self.time]["close"]
        # エントリーしてからいくつバーが進んだか
        if self.MarketPosition != 0:
            self.BarsSinceEntry += 1
            if self.MaxHoldCount < self.BarsSinceEntry:
                self.MaxHoldCount = self.BarsSinceEntry
        # 約定処理
        if self.OrderPrice == 0 or self.OrderPosition == 0:
            return
        if self.MarketPosition == 0 and self.OrderPosition == 1:
            if self.l <= self.OrderPrice:
                self.MarketPosition = 1
                self.BarsSinceEntry = 0
                self.EntryTime = self.time
                self.EntryPrice = self.OrderPrice
                self.StockUnit = self.OrderUnit
                self.LECount += 1
                self.log(
                    f"{self.time} {self.df.iloc[self.time]['timestamp']} AI LE {self.profit}"
                )
        if self.MarketPosition == 0 and self.OrderPosition == -1:
            if self.OrderPrice <= self.h:
                self.MarketPosition = -1
                self.BarsSinceEntry = 0
                self.EntryTime = self.time
                self.EntryPrice = self.OrderPrice
                self.StockUnit = self.OrderUnit
                self.SECount += 1
                self.log(
                    f"{self.time} {self.df.iloc[self.time]['timestamp']} AI SE {self.profit}"
                )
        
        self.OrderUnit = 0
        self.OrderPrice = 0
        self.OrderPosition = 0
    
    def step(self, action):
        self.info = {}
        reward = 0
        miss_reward = -1
        miss = False
        done = self.time == (len(self.df) - 1)

        # 現在が安値であると予想
        if action == 1:
            # 現物買い
            if self.MarketPosition == 0:
                self.LongEntry(self.c)
                reward = self.clairvoyants()
            # 買い戻し
            if self.MarketPosition == -1:
                if self.c < self.EntryPrice:
                    reward = self.ShortExit(self.c)
                    self.profit += reward
            if self.MarketPosition == 1:
                self.OrderMiss += 1
                reward = miss_reward
        # 現在が高値であると予想
        if action == 2:
            # 信用売り
            if self.MarketPosition == 0:
                self.ShortEntry(self.c)
                reward = self.clairvoyants()
            # 現物売り
            if self.MarketPosition == 1:
                if self.c > self.EntryPrice:
                    reward = self.LongExit(self.c)
                    self.profit += reward
            if self.MarketPosition == -1:
                self.OrderMiss += 1
                reward = miss_reward

        # 予測が外れた場合は成り行き
        if action == 0 or self.FeatureBarsSinceEntry(36) == -1:
            # 買い戻し
            if self.MarketPosition == -1:
                reward = self.ShortExit(self.c)
                self.profit += reward
            # 現物売り
            if self.MarketPosition == 1:
                reward = self.LongExit(self.c)
                self.profit += reward
            # ポジションが無い場合は何もしない 
            if self.MarketPosition == 0:
                self.FlatCount += 1
        if done:
            self.log(f"profit {self.profit}")
        else:
            self.time += 1
        self.info["profit"] = self.profit
        self.info["LE count"] = self.LECount
        self.info["SE count"] = self.SECount
        self.info["exit count"] = self.ExitCount
        self.info["flat count"] = self.FlatCount
        self.info["miss count"] = self.OrderMiss
        self.info["max hold count"] = self.MaxHoldCount
        # 約定処理
        self.ProcessOrder()
        # Next Step Features
        observation = self.get_observation()
        return observation, reward, done, self.info
