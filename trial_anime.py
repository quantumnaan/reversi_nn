# import matplotlib.pyplot as plt
# from matplotlib.animation import ArtistAnimation
# import matplotlib.patches as patches

# y = [0] * 30

# x_data = []
# y_data = []
# artist_list = []

# fig, ax = plt.subplots()

# for index, i in enumerate(y):
#     ax.set_xlim(0, 30)
#     ax.set_ylim(-1, 1)
#     x_data.append(index)
#     y_data.append(i)
#     line = patches.Circle(xy=(0.5, 0.5), radius=0.2, fc='w', ec='b') # line2Dオブジェクトを取得
#     if index==0: print(line)
#     # line = axplot(x_data, y_data) # listオブジェクトとして取得=>[line2Dオブジェクト]
#     artist_list.append([line]) # line2Dをlistオブジェクトとして追加
#     # artist_list.append(line) # listオブジェクトの場合はそのまま渡す


# ani = ArtistAnimation(
#     fig=fig, # 図
#     artists=artist_list, # 出来上がったグラフのリストを渡す
#     interval=100,
# )

# plt.show()

import matplotlib.pyplot as plt

from matplotlib.animation import ArtistAnimation
from matplotlib.animation import PillowWriter, FFMpegWriter



country = ['JPN', 'USA']
population = [131, 301]

# 各値を10飛ばして取得（アニメーションする際の時間短縮として）
jpn_10 = list(range(0, population[0], 10))
usa_10 = list(range(0, population[1], 10))
# 各変数の長さの誤差×[jpn_10の最高値] = 足りない分の値の穴埋め
jpn_10 = jpn_10 + [130] * abs(len(jpn_10) - len(usa_10))

fig = plt.figure(figsize=(10, 5))
fontsize = 15

# 以下は変化しないのでグローバルで定義
plt.title("Population comparison between Japan and the United States", fontsize=fontsize)
plt.xlabel("Country", fontsize=fontsize)
plt.ylabel("Population", fontsize=fontsize)
plt.xlim(-1, 2)
plt.ylim(0, 330)
plt.grid()

# 各オブジェクトが格納されたリストを追加するための空のリストを用意
artist_list = []

for j, u in zip(jpn_10, usa_10):

    # jpバー、usバーインスタンス変数を取得
    bar1, bar2 = plt.bar(country, [j, u], width=0.3, color=['b', 'r'])
    # 他の方法でobjを取得するには以下
    # bars_ojb = plt.bar(country, [j, u], width=0.3, color=['b', 'r'])
    # bars_list = list(bars_obj)

    # デフォルト値として設定
    jp_text = plt.text(0, 0, None, fontsize=fontsize)
    us_text = plt.text(0, 0, None, fontsize=fontsize)

    # mplのテキストオブジェクトを更新
    jp_text.set_position((country[0], j+5))
    jp_text.set_text("{} Millon".format(j))
    us_text.set_position((country[1], u+5))
    us_text.set_text("{} Millon".format(u))

    # 各値のオブジェクトをリストに格納
    # 格納する並びは適当
    objects_list = [bar1, jp_text, bar2, us_text]
    # bars_listを格納する場合
    # objects_list = bars_list + [jp_text, us_text]

    # リストをリストに追加していく
    artist_list.append(objects_list*60)


ani = ArtistAnimation(
    fig=fig, # 図
    artists=artist_list, # 出来上がったグラフのリストを渡す
    interval=100, # 速さを設定（ミリ秒）
)

plt.show()

# gifファイルとして保存する場合
#ani.save("animation_bar.gif", writer=PillowWriter())
# mp4ファイルとして保存する場合
#ani.save("animation_bar.mp4", writer=FFMpegWriter())