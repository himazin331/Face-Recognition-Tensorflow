# -*- coding: utf-8 -*-

# リアルタイム顔認識プログラム(Tensorflow)(開発終了)

import tensorflow as tf
import tensorflow.keras.layers as kl

from PIL import Image
import numpy as np
import cv2

import sys
import os
import argparse as arg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# ==================================== face_recog_train_TF.pyと同じネットワーク構成　====================================
class CNN(tf.keras.Model):
    def __init__(self, n_out):
        super(CNN, self).__init__()
        self.conv1 = kl.Conv2D(16, 5, activation='relu', input_shape=(None, 32, 32, 1))
        self.conv2 = kl.Conv2D(32, 5, activation='relu')
        self.conv3 = kl.Conv2D(64, 5, activation='relu')
        self.mp = kl.MaxPool2D((2, 2), padding='same')
        self.flt = kl.Flatten()
        self.link = kl.Dense(1024, activation='relu')
        self.link_class = kl.Dense(n_out, activation='softmax')

    def call(self, x):
        h1 = self.mp(self.conv1(x))
        h2 = self.mp(self.conv2(h1))
        h3 = self.conv3(h2)
        h4 = self.link(self.flt(h3))
        return self.link_class(h4)
# ======================================================================================================================


def main():
    # プログラム情報
    print("RealTime Face Recognition Program(TF) ver.1")
    print("Last Update Dete:    2020/03/12 (Stop development)\n")

    # コマンドラインオプション引数
    parser = arg.ArgumentParser(description='RealTime Face Recognition Program(Tensorflow)')
    parser.add_argument('--param', '-p', type=str, default=None,
                        help='学習済みパラメータの指定(未指定ならエラー)')
    parser.add_argument('--cascade', '-c', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/haar_cascade.xml'.replace('/', os.sep),
                        help='Haar-cascadeの指定(デフォルト値=./haar_cascade.xml)')
    parser.add_argument('--device', '-d', type=int, default=0,
                        help='カメラデバイスIDの指定(デフォルト値=0)')
    args = parser.parse_args()

    # パラメータファイル未指定時->例外
    if args.param is None:
        print("\nException: Trained Parameter-File not specified.\n")
        sys.exit()
    # 存在しないパラメータファイル指定時->例外
    if os.path.exists(args.param) is False:
        print("\nException: Trained Parameter-File {} is not found.\n".format(args.param))
        sys.exit()
    # 存在しないHaar-cascade指定時->例外
    if os.path.exists(args.cascade) is False:
        print("\nException: Haar-cascade {} is not found.\n".format(args.cascade))
        sys.exit()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Trained Prameter-File: {}".format(os.path.abspath(args.param)))
    print("# Haar-cascade: {}".format(args.cascade))
    print("# Camera device: {}".format(args.device))
    print("===========================")

    # カメラインスタンス生成
    cap = cv2.VideoCapture(args.device)
    # FPS値の設定
    cap.set(cv2.CAP_PROP_FPS, 60)

    # 顔検出器のセット
    detector = cv2.CascadeClassifier(args.cascade)

    # 学習済みパラメータ読の読み込み
    model = CNN(7)
    model.build((None, 32, 32, 1))
    model.load_weights(args.param)

    red = (0, 0, 255)
    green = (0, 255, 0)
    p = (10, 30)

    while True:
        # フレーム取得
        _, frame = cap.read()

        # カメラ認識不可->例外
        if _ is False:
            print("\nException: Camera read failure.\n")
            sys.exit()

        # 顔検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)

        # 顔未検出->continue
        if len(faces) == 0:
            cv2.putText(frame, "face is not found",
                        p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, red, thickness=2)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 顔検出時
        for (x, y, h, w) in faces:
            # 顔領域表示
            cv2.rectangle(frame, (x, y), (x + w, y + h), red, thickness=2)

            # 顔が小さすぎればスルー
            if h < 50 and w < 50:
                cv2.putText(frame, "detected face is too small",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, red, thickness=2)
                cv2.imshow("frame", frame)
                break

            # 検出した顔を表示
            cv2.imshow("gray", cv2.resize(gray[y:y + h, x:x + w], (250, 250)))

            # 画像処理
            face = gray[y:y + h, x:x + w]
            face = Image.fromarray(face)
            face = np.asarray(face.resize((32, 32)), dtype=np.float32)
            face = face / 255
            face = face[np.newaxis, :, :, np.newaxis]
            face = tf.convert_to_tensor(face, np.float32)

            # 顔識別
            y = model.predict(face)
            c = np.argmax(y)

            if c == 0:
                cv2.putText(frame, "Unknown",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 1:
                cv2.putText(frame, "Kohayakawa",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 2:
                cv2.putText(frame, "Kouta",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 3:
                cv2.putText(frame, "Nakayama",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 4:
                cv2.putText(frame, "Saikawa",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 5:
                cv2.putText(frame, "Shion",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
            elif c == 6:
                cv2.putText(frame, "Takei",
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, thickness=2)
                
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
