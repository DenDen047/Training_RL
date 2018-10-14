
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
T_MAX = 10   # 各スレッドの更新ステップ間隔

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


class Environment:
    # env
    # tensorflowのスレッドになる
    total_reward_vec = np.zeros(10) # 総報酬を10試行分格納して，平均総報酬を求める
    count_trial_each_thread = 0 # 各環境の試行回数

    def __init__(self, name, thread_type, parameter_server):
        self.name = name
        self.thread_type = thread_type
        self.env = gym.make(ENV)
        self.agent = Agent(name, parameter_server)  # 環境内で行動するagenetを生成

    def run(self):
        self.agent.brain.pull_parameter_server()    # ParameterServerの重みを自身のLocalBrainにコピー
        global frames   # session全体での試行回数
        global IS_LEARNED

        if self.thread_type is 'test' and self.count_trial_each_thread == 0:
            self.env.reset()
            self.env = gym.wrappers.Monitor(self.env, './movie/A3C')    # 動画保存する場合

        s = self.env.reset()
        r_sum = 0
        step = 0
        while True:
            if self.thread_type is 'test':
                self.env.render()   # 学習後のテストではrenderingする
                time.sleep(0.1)
            
            a = self.agent.act(s)   # actionを決定
            s_, r, done, info = self.env.step(a)    # 行動を実施
            step += 1
            frames += 1 # セッション全体の行動回数をカウント

            r = 0
            if done:    # terminal state
                s_ = None
                if step < 199:
                    r = -1
                else:
                    r = 1

            # Advantageを考慮したrewardと経験を，localBrainにpush
            self.agent.advantage_push_local_brain(s, a, r, s_)

            s = s_
            r_sum += r
            # 終了時がT_MAXごとに，parameterServerのweightを更新し，それをコピーする
            if done or step % T_MAX == 0:    
                if not IS_LEARNED and self.thread_type is 'train':
                    self.agent.brain.update_parameter_server()
                    self.agent.brain.pull_parameter_server()
            
            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], step))    # 合計報酬の古いものを削除して，最新の10個を保持
                self.count_trial_each_thread += 1   # このスレッドの総試行回数を増やす
                break
        
        # 学習結果を表示
        print('スレッド: {}\t試行回数: {}\t今回のステップ: {}\t平均ステップ: {}'.format(
            self.name, self.count_trial_each_thread, step, self.total_reward_vec.mean()
        ))

        # スレッドで平均報酬が一定を超えたら終了
        if self.total_reward_vec.mean() > 199:
            IS_LEARNED = True
            time.sleep(2.0) # この時間で，他のtrain threadが止まる
            self.agent.brain.push_parameter_server()    # 成功したスレッドのパラメータをparameterSereverにわたす





if __name__ == "__main__":
    # execute only if run as a script
    main()