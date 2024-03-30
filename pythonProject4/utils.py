import numpy as np
import pandas as pd
import json
import swifter


def generate_safd(df):
    df_ = df.copy()
    df_['acceleration'] = df_['SPEED'].diff(1) / 3.6  # 转化为m/s2
    df_index = df_['acceleration'].copy()[(df_['acceleration'] < 0) & (df_['acceleration'] > -0.1)].index
    df.loc[df_index, 'acceleration'] = 0
    df_['acceleration'] = round(df_['acceleration'], 1)
    df_['SPEED'] = round(df['SPEED'], 0)
    df_.fillna(0.0, inplace=True)  # 下一步，将其转化为speed, acceleration矩阵
    mask = df_['SPEED'] == 0
    df_.loc[mask, 'acceleration'] = 0
    Frequency_df = df_.groupby(['acceleration', "SPEED"]).size().reset_index(name='Frequency')
    Frequency_df["Frequency"] = Frequency_df["Frequency"] / df.shape[0]
    return Frequency_df


def cal_total_loss(df, total_frequency_df):
    """
    :return: 计算和原分布的差，损失函数
    """
    merged_df = pd.merge(df, total_frequency_df, how="right", left_on=["SPEED", "acceleration"],
                         right_on=["SPEED", "acceleration"])
    merged_df.fillna(0, inplace=True)
    return np.sum(np.abs(merged_df["Frequency_x"] - merged_df["Frequency_y"]))


def generate_new_characters(df,total_frequency_df):
    """
    :param df: 输入的df
    :return: 14个特征的东西
    """
    df_road_power = (86.3 * df["SPEED"] + 0.0459 * (df["SPEED"]) ** 3) / 1000 # 计算基本的 road power，不考虑 acceleration
    df_road_power[df["acceleration"] > 0] += (317 * df["acceleration"] * df["SPEED"]) / 1000 # 对于 acceleration > 0 的情况，更新 road power 计算
    safd_df = generate_safd(df)
    SAFD = cal_total_loss(safd_df,total_frequency_df)
    avg_road_power = df_road_power.mean()
    avg_speed = df['SPEED'].mean()
    avg_running_speed = df['SPEED'][df['SPEED'] != 0].mean()
    max_speed = df['SPEED'].max()
    avg_positive_acceleration = df['acceleration'][df['acceleration'] > 0].mean()
    avg_negative_acceleration = df['acceleration'][df['acceleration'] < 0].mean()
    rms_acceleration = np.sum(np.power(df['acceleration'], 2)) / len(df)
    idling_speed_ratio = df["SPEED"][df['SPEED'] == 0].shape[0] / len(df)
    accelerate_time_ratio = df["acceleration"][df['acceleration'] > 0].shape[0] / len(df)
    deceleration_time_ratio = df["acceleration"][df['acceleration'] < 0].shape[0] / len(df)
    cruise_time_ratio = df.query('SPEED >= 5 and acceleration >= -0.1 and acceleration <= 0.1').shape[0] / len(df)
    creep_time_ratio = df.query('SPEED < 5 and acceleration >= -0.1 and acceleration <= 0.1').shape[0] / len(df)
    result = {"avg_road_power": avg_road_power,"avg_speed": avg_speed,"avg_running_speed": avg_running_speed,"max_speed": max_speed,"avg_positive_acceleration": avg_positive_acceleration,
        "avg_negative_acceleration": avg_negative_acceleration,"rms_acceleration": rms_acceleration,"idling_speed_ratio": idling_speed_ratio,"accelerate_time_ratio": accelerate_time_ratio,
        "deceleration_time_ratio": deceleration_time_ratio,"cruise_time_ratio": cruise_time_ratio,"creep_time_ratio": creep_time_ratio,"SAFD":SAFD}
    for key in result:
        if np.isnan(result[key]):
            result[key] = 0
    return result


if __name__ == '__main__':
    df = pd.read_csv(r'.\intermediate_df\data.csv')
    generate_safd(df).to_csv(r'./Frequency/frequency.csv')
    # df = pd.read_csv(r'.\test\test.csv')
    total_frequency_df = pd.read_csv(r'.\Frequency\frequency.csv')
    info_json = json.dumps(generate_new_characters(df, total_frequency_df), sort_keys=False, indent=4, separators=(',', ': '))
    with open('info.json', 'w') as f:
        f.write(info_json)
        f.close()

