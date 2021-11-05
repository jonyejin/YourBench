# YourBench - Pytorch

<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

YourBench�� ������� ���� �Է¹޾� adversarial attack �� �󸶳� robust���� �򰡸� ���ִ� pytorch library�Դϴ�. �� �����ڵ��� ���� ���� adversarial training�� �� �� �ֵ��� ���� �� ��ǥ�� Report�� �Բ� �����մϴ�.

## ����

1. [����](#����)
2. [�����](#�����)
3. [���ɺ�](#���ɺ�)
4. [Contribution](#contribution)
5. [�������](#�������)



## ����

### :mega: ������ �����̶�?
������ ������ ������ ���� �����ϴ� ���� ��ǥ���� ����Դϴ�. ������ ���� �н���Ű�� ����� ������ �� ���ݿ� �̿��Ͽ� ���� �ùٸ� ������ ���� ���ϵ��� ������ �� �ֽ��ϴ�. �ΰ��� ������ �Ȱ��� ������������, �𵨿� �Է��ϸ� ���� �ٸ� ����� ���� �� �ִ� ������. **���� �׽�Ʈ �̹����� �� �з��ϴ���, �̷��� ������ ���ݿ� ����ϴٸ� ����ϱ� ����� ���Դϴ�.**


### :pencil2: ������ �н��� �߿伺
���� test data�� ���ؼ� ����� �ŷڼ� �ִ� ����� ������, ������ ������ ���ۿ� ����ϴٸ� ���� �� �� ���Ե˴ϴ�. adversarial attack�� model robustness�� ������ ���� �����Դϴ�. ���� ������ �����ϸ鼭 ������������ �ϱ� �����Դϴ�. ���� �ڽ��� �Ű��, �Ǵ� ���� adversarial attack�� ���ؼ� robust ������, �������� ���ο� ���� ����� ��Ÿ�� �� �ֽ��ϴ�. ���� �� ������ ���忡�� ���ο� ���ݱ���� ���ؼ� �� ����ϴ� �ڼ��� �߿��մϴ�. **�ٸ� �� ���� �ð��� ���� ��� ������ �ڽ��� �Ű���� ������� �˷��� �ִ� ������ adversarial attack�� �󸶳� robust���� Ȯ���ϴ� ���μ��� ���� �߿��ϴٰ� �� �� �ֽ��ϴ�.**

### :bulb: ���̺귯���� ����
�ٸ� ���̺귯���ʹ� �޸�, YourBench�� ���� Neural Network�� �Է¹޾� adversarial attack�� ���� Benchmark ������ report�� �Բ� �����մϴ�. Report�� ���� ������� �׿� ���� ���� ����� �����մϴ�.�����ڴ� �̸� ���ؼ� �ڽ��� ���� �������� ������� ���� ������ �� ���� ���Դϴ�. 


## �����

```
pip install yourbench
```

###  :warning: �������
YourBench�� ���� ��Ȯ�� test�� �����ϰ� report�� �����ϱ� ���ؼ� ���� ������ �𵨿� ���ؼ� ��������� �Ӵϴ�.
* **No Zero Gradients** \
Obfuscated Gradients�� �˷��� Vanishing/Exploding gradients, Shattered Gradients, �׸��� Stochastic Gradients�� ����ϴ� �𵨿� ���ؼ��� ����� �������� �ʽ��ϴ�. �� gradients�� ����ϴ� ���� ������ ��� ����� �ƴϸ�, adversarial attack generation�� �ſ� ����ϴ�. Obfuscated gradients�� ����ϴ� �𵨵��� EOT�� BPDA, Reparameterizing�� ���� �����ϴ� ���� �����մϴ�.
* **No Loops in Forward Pass** \
Forward pass�� loop�� �ִ� ���� backpropagation�� ����� ������Ű��, �ð��� ���� �ɸ��� �մϴ�. �̷��� �𵨵鿡 ���ؼ� loop�� loss�� �ش� ���� task�� ���Ͽ� ���������� ������ �� �ִ� ������ �����մϴ�.

### :rocket: ����

* ���� �𵨷� �����ϱ�
```python
import yourbench
atk = yourbench.PGD(model, eps=8/255, alpha=2/255, steps=4)
adv_images = atk(images, labels)
```
* ����ڷκ��� �����ͼ°� ��, �׸��� �̹��� ���̺��� �Է¹޾� ������ ������ �� �ֽ��ϴ�.

```python
# label from mapping function
atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:(labels+1)%10)
```

* Strong attacks
```python
atk1 = torchattacks.FGSM(model, eps=8/255)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* Binary serach for CW
```python
atk1 = torchattacks.CW(model, c=0.1, steps=1000, lr=0.01)
atk2 = torchattacks.CW(model, c=1, steps=1000, lr=0.01)
atk = torchattacks.MultiAttack([atk1, atk2])
```

* ���� Report 
</br>\
![adversarial attack-17](https://user-images.githubusercontent.com/80820556/136702057-26e82c95-8536-4619-b9b1-d8cda12d9c55.jpg)

</p></details>


### :fire: �����ϴ� ������ ������ �ο� ��

|          Name          | Paper                                                        | Remark                                                       |
| :--------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  **FGSM**<br />(Linf)  | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |                                                              |                   |
|    **CW**<br />(L2)    | Towards Evaluating the Robustness of Neural Networks ([Carlini et al., 2016](https://arxiv.org/abs/1608.04644)) |                                                              |                    |
|  **PGD**<br />(Linf)   | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083)) | Projected Gradient Method                                    |
| **DeepFool**<br />(L2) | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599)) |                               |



## ���ɺ�

�𵨿� ���� ������ �ŷڵ��� ��� ���ؼ� [Robustbench](https://github.com/RobustBench/robustbench) �� �ο��մϴ�. 


|  **Attack**  |     **Package**     |     Standard |     [Wong2020Fast](https://arxiv.org/abs/2001.03994) |     [Rice2020Overfitting](https://arxiv.org/abs/2002.11569) |     **Remark**     |
| :----------------: | :-----------------: | -------------------------------------------: | -------------------------------------------: | ---------------------------------------------: | :----------------: |
|      **FGSM** (Linf)      |    Torchattacks     | 34% (54ms) |                                 **48% (5ms)** |                                    62% (82ms) |                    |
|  | **Foolbox<sup>*</sup>** | **34% (15ms)** |                                     48% (8ms) |                  **62% (30ms)** |                    |
|                    |         ART         | 34% (214ms) |                                     48% (59ms) |                                   62% (768ms) |                    |
| **PGD** (Linf) |    **Torchattacks** | **0% (174ms)** |                               **44% (52ms)** |            **58% (1348ms)** | :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% (354ms) |                                  44% (56ms) |              58% (1856ms) |                    |
|                    |         ART         | 0% (1384 ms) |                                   44% (437ms) |                58% (4704ms) |                    |
| **CW<sup>��?</sup>**(L2) |    **Torchattacks** | **0% / 0.40<br /> (2596ms)** |                **14% / 0.61 <br />(3795ms)** | **22% / 0.56<br />(43484ms)** | :crown: **Highest Success Rate** <br /> :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.40<br /> (2668ms) |                   32% / 0.41 <br />(3928ms) |                34% / 0.43<br />(44418ms) |  |
|                    |         ART         | 0% / 0.59<br /> (196738ms) |                 24% / 0.70 <br />(66067ms) | 26% / 0.65<br />(694972ms) |  |
| **PGD** (L2) |    **Torchattacks** | **0% / 0.41 (184ms)** |                  **68% / 0.5<br /> (52ms)** |                  **70% / 0.5<br />(1377ms)** | :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.41 (396ms) |                       68% / 0.5<br /> (57ms) |                     70% / 0.5<br /> (1968ms) |                    |
|                    |         ART         | 0% / 0.40 (1364ms) |                       68% / 0.5<br /> (429ms) | 70% / 0.5<br /> (4777ms) |                           |

<sup>*</sup> FoolBox�� ��Ȯ���� adversarial image�� ���ÿ� ��ȯ�ϱ� ������ ���� image generation �ð��� ����� �ͺ��� ª�� �� �ֽ��ϴ�.


## Contribution
### :star2: Contribution �� ������ ȯ���Դϴ�. 
Adversarial Attack�� �����ε� ��� ���Ӱ� ���� ���Դϴ�. YourBench�� ���Ŀ��� model�� adversarial robustness�� ������ ��  ���Ǵ� ���̺귯���� �� �� �ֵ��� ����ϰ��� �մϴ�. ������ ���ο� adversarial attack�� ���´ٸ� �˷��ּ���! YourBench �� contribute �ϰ� �ʹٸ� �Ʒ��� �������ּ���.
[CONTRIBUTING.md](CONTRIBUTING.md).



##  �������

* **Adversarial Attack Packages:**
  
    * [https://github.com/IBM/adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox): Adversarial attack and defense package made by IBM. **TensorFlow, Keras, Pyotrch available.**
    * [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox): Adversarial attack package made by [Bethge Lab](http://bethgelab.org/). **TensorFlow, Pyotrch available.**
    * [https://github.com/tensorflow/cleverhans](https://github.com/tensorflow/cleverhans): Adversarial attack package made by Google Brain. **TensorFlow available.**
    * [https://github.com/BorealisAI/advertorch](https://github.com/BorealisAI/advertorch): Adversarial attack package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust): Adversarial attack (especially on GNN) package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * https://github.com/fra31/auto-attack: Set of attacks that is believed to be the strongest in existence. **TensorFlow, Pyotrch available.**
    
    
    
* **Adversarial Defense Leaderboard:**
  
    * [https://github.com/MadryLab/mnist_challenge](https://github.com/MadryLab/mnist_challenge)
    * [https://github.com/MadryLab/cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)
    * [https://www.robust-ml.org/](https://www.robust-ml.org/)
    * [https://robust.vision/benchmark/leaderboard/](https://robust.vision/benchmark/leaderboard/)
    * https://github.com/RobustBench/robustbench
    * https://github.com/Harry24k/adversarial-defenses-pytorch
    
    
    
* **Adversarial Attack and Defense Papers:**
  
    * https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html: A Complete List of All (arXiv) Adversarial Example Papers made by Nicholas Carlini.
    * https://github.com/chawins/Adversarial-Examples-Reading-List: Adversarial Examples Reading List made by Chawin Sitawarin.
    * Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard. DeepFool: a simple and accurate method to fool deep neural networks. CVPR, 2016
    * Nicholas Carlini, David Wagner. Toward evaluating the robustness of neural networks. arXiv:1608.04644



* **ETC**:

  * https://github.com/Harry24k/gnn-meta-attack: Adversarial Poisoning Attack on Graph Neural Network.
  * https://github.com/ChandlerBang/awesome-graph-attack-papers: Graph Neural Network Attack papers.
  * https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/README_KOR.md
  * https://sdc-james.gitbook.io/onebook/2.-1/1./1.1.5
  * https://github.com/szagoruyko/pytorchviz
  * https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/README_KOR.md
  * https://www.koreascience.or.kr/article/JAKO202031659967733.pdf
  
