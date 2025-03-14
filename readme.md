1.首先运行dataset.py，生成数据集

2.运行meanstd.py，计算数据集的均值和方差

3.更改train.py中的参数，运行train.py，训练模型

4.eva.py中更改参数，运行eva.py，评估模型

*补充*

1.需要配置环境

2.数据集建议>6000

3.train.py中的ratio设置为1即可

4.batch size，lr，epoch，size需要自己尝试

5.json_to_dataset.py是将json文件转换为数据集的代码，对应labelme或者labelimg标注的json文件

6.image文件夹中存放.jpg文件，mask中存放.png文件