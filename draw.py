import matplotlib.pyplot as plt

x = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
y = [0.00, 82.62, 80.11, 81.54, 89.93, 90.38, 89.49, 94.6, 93.7, 93.09, 95.49]
# 以下两种写法等价，
plt.plot(x, y, 'ro-', color='#4169E1', linewidth=1, alpha = 0.8, label='Not embed')
# plt.flot(x, y, 'go--'，linewidth=2, markersize=12)
# 可以在一个画布上绘制多张图片，
y1 = [0.00, 80.62, 79.11, 80.54, 88.93, 90.45, 89.76, 94.16, 92.45, 93.78, 95.11]
plt.plot(x, y1, color='red', marker = '*', linestyle='dashed',  linewidth=1, alpha = 0.8, label='Embed')
plt.xlabel("Epochs",family='Times New Roman', weight='black', size=12)
plt.ylabel("Accuracy", family='Times New Roman', weight='black', size=12)
plt.legend(loc='best')
plt.title('Utility & Economy')
plt.show()
