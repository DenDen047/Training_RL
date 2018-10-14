
# coding:utf-8
# -----------------------------------
# OpenGym CartPole-v0 with A3C on CPU
# -----------------------------------
#
# A3C implementation with TensorFlow multi threads.
#
# Made as part of Qiita article, available at
# https://??/
#
# author: Sugulu, 2017

import os

import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

# -- constants of Game
ENV = 'CartPole-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]     # CartPoleは4状態
NUM_ACTIONS = env.action_space.n        # CartPoleは、右に左に押す2アクション
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of LocalBrain
MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE = 5e-3
RMSPropDecaly = 0.99

# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 8   # スレッドの数
Tmax = 10   # 各スレッドの更新ステップ間隔

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 200*N_WORKERS

# 共有変数
IS_LEARNED = False  # 学習が終了したことを示すフラグ

def main():
    frames = 0  # 全スレッドで共有して使用する総ステップ数
    sess = tf.Session() # Tensorflowのセッション開始

    # スレッドを作成する
    with tf.device('/cpu:0'):
        parameter_server = ParameterServer()    # 全スレッドで共有するparamを持つインスタンス
        threads = []    # 並列で走るスレッド
        # 学習用スレッドを準備
        for i in range(N_WORKERS):
            thread_name = 'local_thread_{}'.format(i+1)
            thread.append(
                Worker_thread(
                    thread_name=thread_name,
                    thread_type='train',
                    parameter_server=parameter_server
                )
            )

        # 学習後にテストで走るスレッドを用意
        thread.append(
            Worker_thread(
                thread_name='test_thread',
                thread_type='test',
                parameter_server=parameter_server
            )
        )

    # Tensorflowでマルチスレッドを実行
    coord = tf.train.Coordinator()  # tensorflowでマルチスレッドにするための準備
    sess.run(tf.global_variables_initializer()) # 変数を初期化

    running_threads = []
    for worker in threads:
        job = lambda: worker.run()
        t = threading.Thread(target=job)
        t.start()
        running_threads.append(t)
    
    # スレッドの終了をあわせる
    coord.join(running_threads)


class Worker_thread:
    # スレッドになるクラス
    # スレッドは学習環境environmentを持つ
    def __init__(self, thread_name, thread_type, parameter_server):
        self.environment = Environment(thread_name, thread_type, parameter_server)
        self.thread_type = thread_type
    
    def run(self):
        while True:
            if not(IS_LEARNED) and self.thread_type is 'train': # train threadが走る
                self.environment.run()
            
            if not(IS_LEARNED) and self.thread_type is 'test':  # test threadを止めとく
                time.sleep(1.0)
            
            if IS_LEARNED and self.thread_type is 'train':  # train threadを止めとく
                time.sleep(3.0)

            if IS_LEARNED and self.thread_type is 'test':   # test threadが走る
                time.sleep(3.0)
                self.environment.run()





if __name__ == "__main__":
    # execute only if run as a script
    main()