"""
    默认的是竖值条形图
"""
import numpy as np
import matplotlib.pyplot as plt

# 将全局的字体设置为黑体
plt.rcParams['font.family'] = 'SimHei'

# 数据
N = 5
y = [20, 10, 30, 25, 15]
# z = [[20, 20, 40, 65, 35],
#     [20, 10, 30, 25, 15]]
x = np.arange(N)
# 添加地名坐标
str1 = ("北京", "上海", "武汉", "深圳", "重庆")

# 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
p1 = plt.bar(x, height=y, width=0.2, label="城市指标", tick_label=str1)
# p1 = plt.bar(x, height=z, width=0.5, label="城市卫生", tick_label=str1)

# 添加数据标签，也就是给柱子顶部添加标签
# for a, b in zip(x, y):
#     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

# 添加图例
plt.legend()

# 展示图形
plt.show()