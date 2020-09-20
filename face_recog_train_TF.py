
# 顔画像学習プログラム(Tensorflow)(開発終了)

import argparse as arg
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TFメッセージ非表示

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np

import matplotlib.pyplot as plt

# CNNの定義
class CNN(tf.keras.Model):
    
    # 各層定義
    def __init__(self, n_out):
        super(CNN, self).__init__()
        #　畳み込み層の定義    
        self.conv1 = kl.Conv2D(16, 5, activation='relu', input_shape=(None, 32, 32, 1)) # 1st 畳み込み層
        self.conv2 = kl.Conv2D(32, 5, activation='relu') # 2nd 畳み込み層
        self.conv3 = kl.Conv2D(64, 5, activation='relu') # 3rd 畳み込み層
        
        #　最大プーリング層の定義
        self.mp1 = kl.MaxPool2D((2, 2), padding='same') # 1st 最大プーリング層
        self.mp2 = kl.MaxPool2D((2, 2), padding='same') # 2nd 最大プーリング層
        
        # データ平坦化
        self.flt = kl.Flatten()
        
        #　全ニューロンの線形結合
        self.link = kl.Dense(1024, activation='relu')   # 全結合層
        self.link_class = kl.Dense(n_out, activation='softmax') # クラス分類用全結合層
            
    # フォワード処理
    def call(self, x):   
        
        h1 = self.mp1(self.conv1(x))     # 1st
        h2 = self.mp2(self.conv2(h1))    # 2nd
        h3 = self.conv3(h2)    # 3rd
        
        h4 = self.link(self.flt(h3))    # 全結合層

        # 予測値返却
        return self.link_class(h4)  # クラス分類用全結合層
        
    # 特徴マップ可視化対応
    def vi_model(self):
        
        x = tf.keras.Input(shape=(32, 32, 1))
        
        return tf.keras.Model(inputs=x, outputs=self.call(x))
    
# Trainer
class trainer(object):
    
    # モデル構築,最適化手法セットアップ
    def __init__(self):
        
        self.model = CNN(7) # モデル構築
        # 最適化手法、損失関数決定
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
        
    # 学習
    def train(self, train_img, train_lab, out_path, batch_size, epochs, fv, mv):

        # 学習
        his = self.model.fit(train_img, train_lab, batch_size=batch_size, epochs=epochs)

        print("___Training finished\n\n")

        # フィルタ可視化
        if fv == True:
            self.filter_vi(self.model)

        # 特徴マップ可視化
        if mv == True:
            self.feature_vi(self.model)

        # パラメータ保存
        print("___Saving parameter...")
        out_path = os.path.join(out_path, "face_recog_tf.h5")
        self.model.save_weights(out_path)
        print("___Successfully completed\n\n")

        return his

    # フィルタ可視化
    def filter_vi(self, model):
        
        vi_layer = []
        
        # 可視化対象レイヤー
        vi_layer.append(model.get_layer('conv2d'))
        vi_layer.append(model.get_layer('conv2d_1'))
        vi_layer.append(model.get_layer('conv2d_2'))
        
        for i in range(len(vi_layer)):      
            
            # 対象レイヤーの重み取得
            target_layer = vi_layer[i].get_weights()[0]
            filter_num = target_layer.shape[3]
            
            # ウィンドウ名定義
            fig = plt.gcf()
            fig.canvas.set_window_title(vi_layer[i].name + " filter visualization")
            
            # プロット
            for j in range(filter_num):
                plt.subplots_adjust(wspace=0.4, hspace=0.8)
                plt.subplot(filter_num/6 + 1, 6, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(f'filter {j}')  
                plt.imshow(target_layer[ :, :, 0, j], cmap="gray") 
            plt.show()

    # 特徴マップ可視化
    def feature_vi(self, model):
        
        # モデルを特徴マップ可視化に対応付け
        vi_model = model.vi_model()
        
        # ネットワーク構成出力
        vi_model.summary()
        print("")
        
        # 各層の定義情報をコピー
        feature_vi = []
        feature_vi.append(vi_model.get_layer('input_1'))
        feature_vi.append(vi_model.get_layer('conv2d'))
        feature_vi.append(vi_model.get_layer('max_pooling2d'))
        feature_vi.append(vi_model.get_layer('conv2d_1'))
        feature_vi.append(vi_model.get_layer('max_pooling2d_1'))

        while 1:
            img = input("画像パスを入力=>")
            if os.path.exists(img) != True:
                print("Exception: Image \"{}\" is not found.\n".format(img))
                continue
            break
        print("")

        # 画像読込, 変換
        img = tf.io.read_file(img)
        img = tf.image.decode_image(img, channels=1) 
        img = img / 255
        img = img[np.newaxis, :, :, :]
        img = tf.convert_to_tensor(img, np.float32)

        for i in range(len(feature_vi)-1):
            
            # モデル実行
            feature_model = tf.keras.Model(inputs=feature_vi[0].output, outputs=feature_vi[i+1].output)
            feature_map = feature_model.predict(img)
            feature_map = feature_map[0]
            feature = feature_map.shape[2]
            
            # ウィンドウ名定義
            fig = plt.gcf()
            fig.canvas.set_window_title(feature_vi[i+1].name + " feature-map visualization")
            
            # プロット
            for j in range(feature):
                plt.subplots_adjust(wspace=0.4, hspace=0.8)
                plt.subplot(feature/6 + 1, 6, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(f'filter {j}')
                plt.imshow(feature_map[:,:,j])
            plt.show()
            
# データセット作成
def create_dataset(data_dir):
    
    print("\n___Creating a dataset...")
    
    cnt = 0
    cnt_t = 0
    prc = ['/', '-', '\\', '|']
    
    # 画像セットの個数
    print("Number of Rough-Dataset: {}".format(len(os.listdir(data_dir))))
    # 画像データの個数
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)
        print("Number of image in a directory \"{}\": {}".format(c, len(os.listdir(d))))
        
    train_img = [] # 画像データ(テンソル)
    train_lab = [] # ラベル
    label = 0
    
    for c in os.listdir(data_dir):

        print("\nclass: {},   class id: {}".format(c, label))   # 画像フォルダ名とクラスIDの出力
        
        d = os.path.join(data_dir, c)                # フォルダ名と画像フォルダ名の結合
        imgs = os.listdir(d)
        
        # JPEG形式の画像データだけを読込
        for i in [f for f in imgs if ('jpg'or'JPG' in f)]:     

            # キャッシュファイルをスルー
            if i == 'Thumbs.db':
                continue

            img = tf.io.read_file(os.path.join(d, i))       #  # 画像フォルダパスと画像パスを結合後、読込
            img = tf.image.decode_image(img, channels=1)    # Tensorflowフォーマットに従ってデコード
            img /= 255     # 正規化

            train_img.append(img)       # 画像データ(テンソル)追加
            train_lab.append(label)     # ラベル追加
            
            cnt += 1
            cnt_t += 1
            
            print("\r   Loading a images and labels...{}    ({} / {})".format(prc[cnt_t%4], cnt, len(os.listdir(d))), end='')
            
        print("\r   Loading a images and labels...Done    ({} / {})".format(cnt, len(os.listdir(d))), end='')
        
        label += 1
        cnt = 0
        
    print("\n___Successfully completed\n")
        
    train_img = tf.convert_to_tensor(train_img, np.float32) # 画像データセット
    train_lab = tf.convert_to_tensor(train_lab, np.int64)   # ラベルデータセット
    
    return train_img, train_lab
    
# 予測精度, 損失値グラフ出力
def graph_output(history):
    
    # 予測精度グラフ
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()  

    # 損失値グラフ
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()


def main():
    
    # プログラム情報
    print("Face Recognition train Program(TF) ver.6")
    print("Last update date:    2020/03/12 (Stop development)\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='Face Recognition train Program(Tensorflow)')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='画像フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='パラメータの保存先指定(デフォルト値=./face_recog_tf.h5')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='ミニバッチサイズの指定(デフォルト値=32)')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='学習回数の指定(デフォルト値=10)')
    parser.add_argument('--f_vi', '-fv', action='store_true',
                        help='フィルタの可視化(指定したら有効)')
    parser.add_argument('--m_vi', '-mv', action='store_true',
                        help='特徴マップの可視化(指定したら有効)')
    parser.add_argument('--g_vi', '-gv', action='store_true',
                        help='予測精度,損失値のグラフ出力(指定したら有効)')
    args = parser.parse_args()

    # 画像フォルダパス未指定->例外
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()
        
    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("\n# Filter visualization: {}".format(args.f_vi))
    print("# Feature-map visualization: {}".format(args.m_vi))
    print("# Function-graph visualization: {}".format(args.g_vi))
    print("===========================")
    
    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)
    
    # データセット作成
    train_img, train_lab = create_dataset(args.data_dir)
    
    # 学習開始
    print("___Start training...")
    Trainer = trainer()
    his = Trainer.train(train_img, train_lab, out_path=args.out, batch_size=args.batch_size, epochs=args.epoch, fv=args.f_vi, mv=args.m_vi)

    # 予測精度, 損失値グラフ出力
    if args.g_vi == True:
        graph_output(his)

if __name__ == '__main__':
    main()