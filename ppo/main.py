# coding:utf-8
# -----------------------------------
# OpenGym CartPole-v0 with PPO on CPU
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
EPS_STEPS = 200 * N_WORKERS

# 共有変数
IS_LEARNED = False  # 学習が終了したことを示すフラグ
FRAMES = 0  # 全スレッドで共有して使用する総ステップ数
SESS = tf.Session() # Tensorflowのセッション開始

def main():
    # スレッドを作成する
    with tf.device('/cpu:0'):
        brain = Brain() # NNのクラス
        threads = []    # 並列して走るスレッド

        # 学習するスレッドを準備
        for i in range(N_WORKERS):
            thread_name = 'train_thread{}'.format(i+1)
            threads.append(Worker_thread(
                thread_name=thread_name,
                thread_type='train',
                brain=brain
            ))

        # 学習後にテストで走るスレッドを準備
        threads.append(Worker_thread(
            thread_name='test_thread',
            thread_type='test',
            brain=brain
        ))

    # Tensorflowでマルチスレッドを実行
    coord = tf.train.Coordinator()  # tensorflowでマルチスレッドにするための準備
    SESS.run(tf.global_variables_initializer()) # 変数を初期化

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
    def __init__(self, thread_name, thread_type, brain):
        self.environment = Environment(thread_name, thread_type, brain)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not IS_LEARNED and self.thread_type is 'train': # train threadが走る
                self.environment.run()

            if not IS_LEARNED and self.thread_type is 'test':  # test threadを止めとく
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

    def __init__(self, name, thread_type, brain):
        self.name = name
        self.thread_type = thread_type
        self.env = gym.make(ENV)
        self.agent = Agent(brain)   # 環境内で行動するagenetを生成

    def run(self):
        global FRAMES   # session全体での試行回数
        global IS_LEARNED

        if self.thread_type is 'test' and self.count_trial_each_thread == 0:
            self.env.reset()
            self.env = gym.wrappers.Monitor(self.env, './movie/PPO')    # 動画保存する場合

        # 環境のリセット
        s = self.env.reset()

        # 各種変数
        r_sum = 0   # 合計報酬
        step = 0    # ループ回数（時刻t）

        # メインループ
        while True:
            if self.thread_type is 'test':
                self.env.render()   # 学習後のテストでは描画
                time.sleep(0.1)

            a = self.agent.act(s)   # actionを決定
            s_, r, done, info = self.env.step(a)    # 行動を実施
            step += 1
            FRAMES += 1 # セッション全体の行動回数をカウント

            r = 0
            if done:    # terminal state
                s_ = None   # doneなのでs+1の状態は不要
                if step < 199:
                    r = -1
                else:
                    r = 1

            # 報酬と経験を，Brainにpush
            self.agent.advantage_push_brain(s, a, r, s_)

            # 情報を更新
            s = s_
            r_sum += r
            # 終了時にTmaxごとに，parameterServerの重みを更新
            if done or FRAMES % T_MAX == 0:
                if not IS_LEARNED and self.thread_type is 'train':
                    self.agenet.brain.update_parameter_server()

            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], step))    # 合計報酬の古いものを削除して，最新の10個を保持
                self.count_trial_each_thread += 1   # このスレッドの総試行回数を増やす
                break

        # 学習結果を表示
        # print(u'スレッド: {}\t試行回数: {}\t今回のステップ: {}\t平均ステップ: {}'.format(
        #     self.name, self.count_trial_each_thread, step, self.total_reward_vec.mean()
        # ))
        print('Thread: {}\tCount Traial: {}\tStep: {}\tAvgStep: {}'.format(
            self.name, self.count_trial_each_thread, step, self.total_reward_vec.mean()
        ))

        # スレッドで平均報酬が一定を超えたら終了
        if self.total_reward_vec.mean() > 199:
            IS_LEARNED = True
            time.sleep(2.0) # この時間で，他のtrain threadが止まる


class Agent(object):
    # 行動を決定するクラス
    def __init__(self, brain):
        self.brain = brain  # 行動を決定するための脳
        self.memory = []    # s,a,r,s_を保存するメモリ
        self.r_sum = 0.     # 時間割引した「今からNステップ後までの」総報酬r_sum

    def act(self, s):
        if FRAMES >= EPS_STEPS: # e-greedy法で行動を決定する
            eps = EPS_END
        else:
            eps = EPS_START + FRAMES * (EPS_END - EPS_START) / EPS_STEPS    # linearly interpolate

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)   # ランダムに行動
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)

            # a = np.argmax(p)    # 最大確率の行動を選択
            a = np.random.choice(NUM_ACTIONS, p=p[0])   # 確率p[0]に従って，行動を選択
            return a

    def advantage_push_brain(self, s, a, r, s_):  # advantageを考慮したs,a,r,s_っをbrainに与える
        def get_sample(memory, n):
            # advantageを考慮し，
            # メモリからnステップ後の状態とnステップ後までのr_sumを取得する関数
            s, a, _, _  = memory[0] # 現在の情報
            _, _, _, s_ = memory[n - 1] # nステップ後の情報
            return s, a, self.r_sum, s_

        # one-hotコーティングしたa_catsを作り，s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬r_sum」を利用して，現ステップのr_sumを計算
        # r0は後で引き算している
        # r0には取り出したい期間以前の報酬情報を含んでいる
        self.r_sum = (self.r_sum + r * GAMMA_N) / GAMMA

        # advantageを考慮しながら，Brainに経験を入力する
        if s_ is None:  # done=1 状態
            # doneしているので，残っているmemoryをBrainにpush
            while len(self.memory) > 0:
                # nステップ後のパラメータを記録
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                # 次のループに備え，r0を引く
                self.r_sum = (self.r_sum - self.memory[0][2]) / GAMMA
                self.memory.pop(0)  # 指定した位置の要素を削除し、値を取得

            self.r_sum = 0  # 次の試行に向けてリセットしておく

        if len(self.memory) >= N_STEP_RETURN：
            # 十分に情報が溜まった状態で実行
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.r_sum -= self.memory[0][2] # r0を引き算
            self.memory.pop(0)


class ParameterServer:
    # グローバルなtensorflowのDNNのクラス
    def __init__(self):
        # スレッド名で重み変数に名前を与えて，識別している
        with tf.variable_scope('parameter_server'):
            self.model = self._build_model()

        # serverのパラメータ宣言
        self.weights_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='parameter_server')
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSPropDecaly)    # loss関数を最小化していくoptimizerの定義

    def _build_model(self):
        # kerasでネットワークを定義
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])

        # # Qネットワークを可視化
        # plot_model(model, to_file='A3C.png', show_shapes=True)

        return model


class Brain:
    def __init__(self):
        # globalなparameter_serverをメンバ変数として持つ
        with tf.name_scope('brain'):
            self.train_queue = [[] for i in range(5)]   # s, a, r, s', s' terminal mask
            K.set_session(SESS)
            self.model = self._build_model()    # NNを生成
            self.opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)  # loss関数を最小化していくoptimizerの定義
            self.prob_old = 1
            self.graph = self._build_graph()    # NNの学習やメソッドを定義

    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        plot_model(model, to_file='PPO.png', show_shapes=True)
        return model

    def _build_graph(self):
        # tensorflowでNNの重みをどう学習させるかを定義
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = self.model(self.s_t)

        # loss関数を定義
        advantage = tf.subtract(self.r_t, v)
        self.prob = tf.multiply(p, self.a_t) + 1e-10
        r_theta = tf.div(self.prob, self.prob_old)
        advantage_CPI = tf.multiply(r_theta, tf.stop_gradient(advantage))

        # CLIPした場合を計算して，小さい方を使用する
        r_clip = r_theta
        r_clip = tf.clip_by_value(
            r_clip,
            clip_value_min=r_theta - EPSILON,
            clip_value_max=r_theta + EPSILON
        )

        clipped_advantage_CPI = tf.multiply(
            r_clip, tf.stop_gradient(advantage))
        loss_CLIP = -tf.reduce_mean(
            tf.minimum(advantage_CPI, clipped_advantage_CPI),
            axis=1,
            keep_dims=True)

        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(
            p * tf.log(p + 1e-10), axis=1, keep_dims=True
        )   # maximize entropy(regularization)

        self.loss_total = tf.reduce_mean(loss_CLIP + loss_value - entropy)

        # 求めた勾配で重み変数を更新する定義
        minimize = self.opt.minimize(self.loss_total)
        return minimize

    def update_parameter_server(self):
        # Brainの勾配でParameterServerの重みを学習・更新
        if len(self.train_queue[0]) < MIN_BATCH:
            # データが溜まっていない場合は更新しない
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[] for i in range(5)]
        s = np.vstack(s)    # 行列の転置
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        # Nステップ後の状態s_から，その先で得られるであろう時間割引総報酬v
        _, v = self.model.predict(s_)

        # N-1ステップ後までの時間割引総報酬rに，
        # Nから先に得られるであろう総報酬vに割引N乗したものを足す
        r += GAMMA_N * v * s_mask   # set v to 0 where s_ is terminal state
        feed_dict = {
            self.s_t: s,
            self.a_t: a,
            self.r_t: r
        }   # 重みの更新に使用するデータ

        minimize = self.graph
        SESS.run(minimize, feed_dict)   # Brainの重みを更新
        self.prob_old = self.prob

    def predict_p(self, s):
        # 状態sから各actionの確率ベクトルpを返す
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:  # done=1
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:   # done=0
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)


if __name__ == "__main__":
    # execute only if run as a script
    main()
