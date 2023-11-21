import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import xticks,yticks,np
#导入实验数据
df1 = pd.read_csv(r"D:\yolov5_new\results\yolov5.csv")
df2 = pd.read_csv(r"D:\yolov5_new\results\yolov5Imp.csv")


#查看列名，因为有的会有空格等
print(df1.columns)
#比如我自己的输出结果是这样的，如果map不复制这个可能会导致下面程序报错
'''
out=Index(['               epoch', '      train/box_loss', '      train/obj_loss',
       '      train/cls_loss', '   metrics/precision', '      metrics/recall',
       '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',
       '        val/obj_loss', '        val/cls_loss', '               x/lr0',
       '               x/lr1', '               x/lr2'],
      dtype='object')
'''
#选取作为y轴的数据，这里可以选择你想选的列，比如我这里都是300轮，每一列都会是300个数据。
#但是由于模型早停机制会出现小于300的，这时你可以用下面方法统一设置，如果不需要删除[:297]即可。
# df11 = df1['     metrics/mAP_0.5'][:199]
# df12 = df2['     metrics/mAP_0.5'][:199]
df11 = df1['metrics/mAP_0.5:0.95'][:199]
df12 = df2['metrics/mAP_0.5:0.95'][:199]

plt.figure(figsize=(10,8), dpi=400)  #dpi是分辨率，越高清晰度越高

x = [i for i in range(0,199)] #创建x轴
y1 = df11  #创建y轴
y2 = df12


#plt.title('各模型mAP@0.5曲线')  # 标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.rcParams['axes.unicode_minus'] =False

plt.xlabel('epoch',fontsize=20)  # x轴标题以及标题大小设置
plt.ylabel('mAP@0.5:0.95',fontsize=20)  # y轴标题
#刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=15)
plt.xticks(np.linspace(0,200,9,endpoint=True))
# 修改纵坐标的刻度
plt.yticks(np.linspace(0,0.9,9,endpoint=True))
plt.plot(x, y1)  # 绘制折线图
plt.plot(x, y2)

# 设置曲线名称
plt.legend(['YOLOv5(Baseline)', 'YOLOv5-EOW'],loc=0,fontsize='xx-large')
#图例大小可选----'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'

plt.savefig('mAP@0.5.png',bbox_inches='tight',pad_inches=0) #保存图片，这里增加这两个参数可以消除保存下来图像的白边节省空间，bbox_inches='tight',pad_inches=0)
plt.show()  # 显示曲线图
'''
注意，这里加上plt.show()后，保存的图片就为空白了，因为plt.show()之后就会关掉画布，
所以如果要保存加显示图片的话一定要将plt.show()放在plt.savefig(save_path)之后
'''
