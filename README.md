# reversi_nn
a bot with NN playing reversi game

2023/7/2更新
- reversi_data_collect.py:
    モンテカルロ法を用いて(盤面，各マスに打った時の勝率)というデータを自動で集めてrecord/win_prob.pklに保存する
- model.pth:
    学習したモデルを保存しておく場所．reversi_nn.pyでload/saveを行う
- computer.py:
    モンテカルロ法のメソッドをまとめてあるところ
- reversi_env.py:
    盤面の可視化など．このプログラムで人間と対戦出来る．対戦にモンテカルロ法を用いた場合はデータはrecord/win_prob.pklに保存する
- reversi_nn.py:
  　neural network，強化学習アルゴリズム，データのload/saveを行う．
- reversi_env_copied.py:
  　ブログのやつそのまま
