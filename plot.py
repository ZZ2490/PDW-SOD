import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [10, 8, 6, 4, 2]
y2 = [3, 5, 7, 9, 131]

# 创建图形和坐标轴对象
fig, ax1 = plt.subplots()

# 绘制第一个坐标轴的折线图
ax1.plot(x, y1, 'r.-')
ax1.set_xlabel('X轴标签')
ax1.set_ylabel('Y1轴标签', color='r')
ax1.tick_params('y', colors='r')

# 创建第二个坐标轴对象
ax2 = ax1.twinx()

# 绘制第二个坐标轴的折线图
ax2.plot(x, y2, 'g.-')
ax2.set_ylabel('Y2轴标签', color='g')
ax2.tick_params('y', colors='g')
# plt.savefig('1.png')
# 显示图形
plt.show()