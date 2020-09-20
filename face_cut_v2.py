import cv2
import os
import argparse as arg
import sys
import numpy as np
from natsort import natsorted
import mimetypes

# 画像加工
def face_cut(imgs_dir, result_out, img_size, ch, label, HAAR_FILE):

    # Haar-Like特徴量Cascade型分類器の読み込み
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    
    # データ加工
    for img_name in natsorted(os.listdir(imgs_dir)):

        print("画像データ:{}".format(img_name))
        
        # 対応MIMEタイプ -> image
        mime = mimetypes.guess_type(img_name)
        if 'image' in mime[0]:
            
            img = cv2.imread(img_name)  # データ読み込み
            
            # チャンネル数が１であれば
            if ch == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール変換

            face = cascade.detectMultiScale(img)  # 顔を検出
    
            # 顔が検出されなかったら
            if len(face) == 0:
                
                print("Face not found.")
                
            else:
                for n in range(len(face)):
                
                    # X,Y座標、幅/高さ取得
                    x = face[n][0]
                    y = face[n][1]
                    w = face[n][2]
                    h = face[n][3]

                    # 顔の切り取り
                    face_cut = img[y:y+h, x:x+w] 
                    # リサイズ
                    face_img = cv2.resize(face_cut, (img_size, img_size))
        
                    # 保存
                    result_img_name = r'\data' + str(label) + '.jpg'
                    cv2.imwrite(os.path.join(result_out + result_img_name), face_img)
                    label += 1
            
                    # 表示
                    print("Processing success!!")

        else:
            print("Unsupported file extension")
    
def main():

    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='Face image cropping')
    parser.add_argument('--imgs_dir', '-d', type=str, default=None,
                        help='画像フォルダパス(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__))+'/result_crop'.replace('/', os.sep),
                        help='加工後データの保存先(デフォルト値=./reslut_crop)')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='リサイズ(NxNならN,デフォルト値=32)')
    parser.add_argument('--ch', '-ch', type=int, default=1,
                        help='チャンネル数(デフォルト値=1)')
    parser.add_argument('--label', '-l', type=int, default=1,
                        help='dataN.jpgのNの初期値(デフォルト値=1)')
    parser.add_argument('--haar_file', '-c', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/haar_cascade.xml'.replace('/', os.sep),
                        help='haar-Cascadeのパス指定(デフォルト値=./haar_cascade.xml)')
    args = parser.parse_args()

    # 画像フォルダ未指定時->例外
    if args.imgs_dir == None:
        print("\nException: Cropping target is not specified.\n")
        sys.exit()
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.imgs_dir) != True:
        print("\nException: {} does not exist.\n".format(args.imgs_dir))
        sys.exit()     
    # 存在しないCascade指定時->例外
    if os.path.exists(args.haar_file) != True:
        print("\nException: {} does not exist.\n".format(args.haar_file))
        sys.exit()

    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.imgs_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Images size: {}".format(args.img_size))
    print("# Channel: {}".format(args.ch))
    print("# Start index: {}".format(args.label))
    print("# Haar-cascade: {}".format(args.haar_file))
    print("===========================\n")
    
    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    # 加工
    face_cut(args.imgs_dir, args.out, args.img_size, args.ch, args.label, args.haar_file)
    print("")
    
if __name__ == '__main__':
    main()