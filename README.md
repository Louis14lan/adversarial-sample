# 人脸识别对抗攻击
* 相关算法整理资料请参考[reference](https://github.com/Louis14lan/adversarial-sample/blob/master/doc/reference.md)

## 代码
参考了[M-DI2-FGSM算法](https://github.com/ppwwyyxx/Adversarial-Face-Attack),对其做了些改动，将其有目标攻击改为无目标攻击

### 环境准备
* 安装python环境：`pip install -r requirements.txt`
* 下载预训练模型，原作者项目里有提供下载链接

### 无目标攻击
```
python M-DI2-FGSM-nontargeted.py 
```