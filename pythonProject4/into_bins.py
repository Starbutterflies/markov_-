import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import generate_new_characters
import random
import json


class into_bins(object):
    def __init__(self):
        self.markov_array = None
        self.labels = None
        self.origin_data = pd.read_csv(r'./intermediate_df/data.csv')
        self.origin_data["diff"] = self.origin_data["label"].diff()  # 读取数据
        self.origin_data.fillna(0, inplace=True)
        self.seg_list = [[] for i in range(6)]
        self.character_list = []
        self.ordered_label = []
        self.total_frequency_df = pd.read_csv(r"./Frequency/frequency.csv")
        with open("info.json","r") as file:
            str_ = file.read()
            file.close()
        self.database_character = pd.DataFrame([json.loads(str_)])
        self.seg_list_markov = []

    def into_bins(self):
        """
        :return:切成小块的bin
        """
        slice_ = self.origin_data[self.origin_data["diff"] != 0].index  # 切割点位
        for i in tqdm(range(len(slice_) - 1)):
            if slice_[i + 1] - slice_[i] == 1:
                df = self.origin_data.loc[slice_[i]]
                label = df['label']
                self.seg_list[int(label)].append(df)
            else:
                df = self.origin_data.loc[slice_[i]:slice_[i + 1] - 1]
                label = df['label'].unique()
                self.seg_list[int(label)].append(df)


    def generate_characters(self, df):
        """
        :param df:
        :return:
        """
        length = len(df)
        if length == 1:
            average_speed = df["SPEED"]
            maximum_speed = df["SPEED"]
            minimum_speed = df["SPEED"]
            if df["acceleration"] > 0:
                acceleration_rates = 1
            elif df["acceleration"] <= 0:
                acceleration_rates = 0
        else:
            average_speed = df["SPEED"].mean()
            maximum_speed = df["SPEED"].max()
            minimum_speed = df["SPEED"].min()
            acceleration_rates = len(df["acceleration"][df["acceleration"] > 0]) / length
        return average_speed, maximum_speed, minimum_speed, acceleration_rates

    def generate_K(self, df):
        """
        :return: 这个还是只能看啊
        """
        SSE = []
        for k in range(1, 15):
            estimator = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)  # 构造聚类器
            estimator.fit(df)
            SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
            print("完成一次循环！")
        X = range(1, 15)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(X, SSE, 'o-')
        plt.show()
        series = pd.Series(SSE).diff(-1) / pd.Series(SSE).diff(-1).sum()
        print(series)

    def apply_characters_cluster(self):
        """
        :return:
        """
        segment_list = [df for dfs in self.seg_list for df in dfs]
        for df in segment_list:
            self.character_list.append(self.generate_characters(df))
        character_df = pd.DataFrame(self.character_list,
                                    columns=["average_speed", "maximum_speed", "minimum_speed", "acceleration_rates"])
        transformer = StandardScaler()
        character_df = transformer.fit_transform(character_df)
        # self.generate_K(character_df)
        estimator = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300)
        estimator.fit(character_df)
        self.labels = estimator.labels_  # 进行分类

    def generate_markov_array_1d(self):
        """
        :return:生成的一维马尔科夫矩阵，按照平均速度划分状态
        """
        seg_list = [df for dfs in self.seg_list for df in dfs]
        sum_array = np.zeros(max(self.labels) + 1, dtype=float)
        len_array = np.zeros(max(self.labels) + 1, dtype=float)
        for label, df in list(zip(self.labels, seg_list)):
            sum_array[label] += df["SPEED"].sum()
            len_array[label] += len(df)
        avg_speed = sum_array / len_array
        new_rank = np.argsort(np.argsort(avg_speed))
        trans_ = dict(zip([i for i in range(len(new_rank))], new_rank))
        for label in self.labels:
            self.ordered_label.append(new_rank[label])
        self.markov_array = np.zeros([6, 6], dtype=np.int64)
        for i in range(len(self.labels) - 1):
            self.markov_array[self.ordered_label[i], self.ordered_label[i + 1]] += 1
        self.markov_array = self.markov_array / np.sum(self.markov_array, axis=0)
        print(self.markov_array)

    def generate_driving_cycle_1d(self):
        """
        :return: 生成的新工况
        """
        old_seg_list = [seg for list_ in self.seg_list for seg in list_]  # 修正标签
        new_seg_list = [[] for i in range(6)]
        for (seg, label) in zip(old_seg_list, self.ordered_label):  # 用于修正df和series
            if len(seg.shape) != 1:
                seg_ = seg.copy()
                seg_.loc[:,"label"] = np.full(seg.shape[0], label)
                new_seg_list[label].append(seg_)
            else:
                seg_ = seg.to_frame().T
                seg_.loc[:,"label"] = label
                seg_.columns = ['SPEED', 'acceleration', 'label', 'diff']
                new_seg_list[label].append(seg_)

        start_list = [[df for df in seg_list if df["SPEED"].iloc[0] == 0 and df["SPEED"].iloc[-1] > 2 and (df["SPEED"][df["SPEED"]>0].shape[0]/df.shape[0]) > 0.5 and len(df) < 100] for seg_list in new_seg_list]
        seg = random.choice(start_list[0])  # 选取开始片段，开始迭代。先选取一个初始的seg
        self.seg_list_markov.append(seg)
        next_stage = 0
        new_speed = seg["SPEED"].iloc[-1]

        while len(seg) <= 1800:
            next_stage = random.choices(np.arange(6), weights=self.markov_array[next_stage])[0]
            print(self.markov_array[next_stage])
            print(next_stage)
            new_list = [df for df in new_seg_list[next_stage] if (np.abs(df["SPEED"].iloc[0]-new_speed)<=2.5 and len(df) <= 100)]
            new_character_list = [generate_new_characters(pd.concat([seg,df]).reset_index(drop=True),self.total_frequency_df) for df in new_list]
            new_df = pd.DataFrame(new_character_list)  # 然后与json相减

            if len(new_list) == 0:
                print("*" *100)
                print("当前状态",next_stage)
                print("没有对应的东西亚撒西")
                print("*" * 100)
                continue
            else:
                print("当前状态：",next_stage)
                database_character = pd.DataFrame(self.database_character.values.repeat(new_df.shape[0], axis=0), columns=self.database_character.columns)
                new_id = np.average(np.abs(new_df-database_character).rank(),axis=1).argmin()  # 得到最为合适的片段id
                new_seg = new_list[new_id]
                self.seg_list_markov.append(new_seg)
                seg = pd.concat([seg,new_seg]).reset_index(drop=True)
                new_speed = seg["SPEED"].iloc[-1]
            print(len(seg))
            print(seg)


if __name__ == '__main__':
    Bin = into_bins()
    Bin.into_bins()
    Bin.apply_characters_cluster()
    Bin.generate_markov_array_1d()
    Bin.generate_driving_cycle_1d()
