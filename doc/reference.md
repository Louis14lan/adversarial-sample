# 人脸识别对抗攻击
## 网络相关文章
[知乎对抗攻击专栏](https://zhuanlan.zhihu.com/c_170476465)<br/>
[2019cvpr对抗汇总**](https://zhuanlan.zhihu.com/p/78743205)<br>
[一篇很好的对抗综述性论文](https://arxiv.org/pdf/1712.07107.pdf)

## 步骤
* 研究相关算法
* 研究常见神经网络模型及代码框架


## 攻击算法综述（相关论文）
#### 基于梯度（白盒）
* FGSM(Fast gradient sign method)：
[[1]](https://blog.csdn.net/qq_35414569/article/details/80770121)指出了之所以存在对抗样本，是因为神经网络模型在高维空间中具有线性特征，并且介绍一种对抗样本生成方法FGSM
* IFGSM(Iterative Fast gradient sign Method)：
[[1]](https://arxiv.org/pdf/1607.02533.pdf)介绍了三种对抗样本生成方法，分别是FGSM、IFGSM、Iterative least-like FGSM
* MI-FGSM（momentum iterative gradient-based methods ）：
[[1]](https://arxiv.org/pdf/1710.06081.pdf)总结了目前流行的三种攻击算法形式：基于单步的(one-step)、迭代的(Iterative)、优化目标函数(Optimization)。提出了基于动量的迭代攻击算法，并且还提出了模型集成的攻击算法。并在nips无目标攻击中取得了第一名成绩。
* DI2-FGSM（Diverse Inputs Iterative Fast Gradient Sign Method）和 M-DI2-FGSM（momentum Diverse Inputs iterative gradient-based methods）：
[[1]](https://arxiv.org/pdf/1803.06978.pdf):总结FGSM算法簇间的不同及联系，提出了对输入先进行随机transform（比如随机resize、padding等），再输入网络进行IFGSM，在一定程度上避免迭代带来的过拟合。
* NRDM（Neural Representation Distortion Method）：
[[1]](https://arxiv.org/pdf/1811.09020.pdf)提出无需loss层与label值，只提取network的某个特征层作为输入的表征，并使其与输入样本的相应表征距离最大化。论文还指出通过实验发现VGG具有较高的可移植性，故本文采用VGG-16 网络的conv3.3 feature map作为网络表征。

#### 基于优化(基本想法就是：让模型预测出label值并计算loss，并对loss值和扰动值进行寻优)
* Deepfool
* C&W(Carlini & Wagner)
* Curls & Whey: Boosting Black-Box Adversarial Attacksation Distortion Method
* Zeroth-Order Optimization（黑盒）
* L-BFGS
[[1]](https://arxiv.org/pdf/1312.6199.pdf):文章论述了可以通过对输入进行细微地扰动达到误导神经网络的效果。本文提出一种扰动机制L-BFGS：建立优化目标，并用BFGS算法求解。


#### 基于几何
* Fast Geometrically-Perturbed Adversarial Faces
[[1]]()
* stAdv(SPATIALLY TRANSFORMED ADVERSARIAL EXAMPLES)
[[1]](https://arxiv.org/pdf/1801.02612.pdf)根据邻近像素值来更新自身像素，更新方法采用优化类，先建立目标模型然后用L-BFGS求解



#### 基于gan
* Perceptual-Sensitive GAN for Generating Adversarial Patches
* Natural Gan

#### 基于决策访问次数：无需特定模型，只需要不断访问目标模型的输出值即可攻击（黑盒）
* Boundary Attack
* Boundary Attack ++
* [Efficient Decision-based Black-box Adversarial Attacks on Face Recognition](https://arxiv.org/pdf/1904.04433.pdf)<br/>
#### 其他
* Universal adversarial perturbations (对所有输入进行统一的扰动达到误导模型的方法)
* One pixel attack for fooling deep neural networks（只对图像中一个像素进行改变）

## 开源代码
### cleverhans（攻击）:
* The Fast Gradient Method attack.
* The Basic Iterative Method attack(BIM)
* The Carlini&Wagner-L2 attack(C&W)
* Deep Fool
* The Elastic Net Method attack
* The Fast Feature Adversaries attack
* The LBFGS attack
* The Madry et al. attack
* The Max Confidence attack
* The Momentum Iterative Method attack
* The Noise attack
* The Projected Gradient Descent attack(PGD)
* The Salience Map Method attack
* The Spatial Transformation Method attack
* The SPSA attack
* The Virtual Adversarial Method attack

### advBox（攻击）
* L-BFGS
* FGSM
* BIM
* ILCM
* MI-FGSM
* JSMA
* DeepFool
* C/W
* Single Pixel Attack
* Local Search Attack

### advBox（防御）
* Feature Fqueezing
* Spatial Smoothing
* Label Smoothing
* Gaussian Augmentation
* Adversarial Training
* Thermometer Encoding

### Foolbox（攻击）：
