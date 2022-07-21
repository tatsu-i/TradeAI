import numpy as np
import heapq
import bisect
import random

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.buffer = []

    def add(self, experience, td_error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, td_error):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)


class _head_wrapper():
    def __init__(self, data):
        self.d = data
    def __eq__(self, other):
        return True

class PERGreedyMemory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience, td_error):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()

        # priority は最初は最大を選択
        priority = abs(td_error)
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))

    def update(self, idx, experience, td_error):
        priority = abs(td_error)
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

#copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

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
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.add(priority, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
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

    def add(self, experience, td_error):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()

        priority = (abs(td_error) + 0.0001)  # priority を計算
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001)  # priority を計算

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
