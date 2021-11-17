# YourBench - Pytorch

<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

  
YourBench is pytorch library, which takes the user's model as input and evaluates how robust it is to adversarial attack. To make it easier for model developers to do adversarial training, the evaluation index of the model is provided along with the Report.  
  
## Contents

1. [Introduction](#Introduction)
2. [How-to-use](#How-to-use)
3. [Performance-Comparison](#Performance-Comparison)
4. [Contribution](#contribution)
5. [reference](#reference)



## Introduction

### :mega: What is adversarial attacks?
Adversarial attacks are the most representative way to attack deep learning models. Conversely, the method of training a deep learning model can be used to attack the model, preventing the model from making correct predictions. It is the same data to the human eye, but when it is input to the model, it can produce completely different results. **Even if the model classifies the test images well, it will be difficult to use if it is vulnerable to such adversarial attacks.**


### :pencil2: The importance of adversarial learning
Even if the model gives sufficiently reliable results on the test data, it becomes unusable if it is vulnerable to simple data manipulation. Adversarial attack and model robustness are the relationship between police and thieves. Because they are constantly evolving and trying to catch up with each other. Even if your current neural network or model is robust against adversarial attacks, new attack techniques may appear at any time. Therefore, it is important to be prepared always for new attack techniques from the model developer's point of view. **However, it is costly and time consuming. So the process of checking how robust your neural network is against known strong adversarial attacks is also important.**

### :bulb: Purpose of the library
Unlike other libraries, YourBench takes personal neural networks as input and provides Benchmark scores for adversarial attacks with report. Report suggests weaknesses in the model and gives way to improve them. Developers can assess the stability of their model through this.. 


## How-to-use

```
pip install yourbench
```

###  :warning: Constraint
YourBench places constraints on the measurable model to perform more accurate tests and provide reports.
* **No Zero Gradients** \
Not recommended for models that use Vanishing/Exploding gradients, Shattered Gradients, and Stochastic Gradients, also known as Obfuscated Gradients. The model using the above gradients is not a suitable defense technique, and adversarial attack generation is very difficult. Models using obfuscated gradients are recommended to attack through EOT, BPDA, and reparameterizing.
* **No Loops in Forward Pass** \
Models with loops in the forward pass increase the cost of backpropagation and take a long time. For these models, we recommend an attack that can be adaptively applied by combining the loop loss and the task of the model.

### :rocket: Demo

* Running with built-in models
```python
import yourbench
atk = yourbench.PGD(model, eps=8/255, alpha=2/255, steps=4)
adv_images = atk(images, labels)
```
* You can take data sets, models, and image labels from the user and do the following.

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

* 예시 Report 
</br>\
![adversarial attack-17](https://user-images.githubusercontent.com/80820556/136702057-26e82c95-8536-4619-b9b1-d8cda12d9c55.jpg)

</p></details>


### :fire: 수행하는 공격의 종류와 인용 논문

|          Name          | Paper                                                        | Remark                                                       |
| :--------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  **FGSM**<br />(Linf)  | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |                                                              |                   |
|    **CW**<br />(L2)    | Towards Evaluating the Robustness of Neural Networks ([Carlini et al., 2016](https://arxiv.org/abs/1608.04644)) |                                                              |                    |
|  **PGD**<br />(Linf)   | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083)) | Projected Gradient Method                                    |
| **DeepFool**<br />(L2) | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599)) |                               |




## Performance-Comparison

모델에 대한 점수의 신뢰도를 얻기 위해서 [Robustbench](https://github.com/RobustBench/robustbench) 를 인용합니다. 


|  **Attack**  |     **Package**     |     Standard |     [Wong2020Fast](https://arxiv.org/abs/2001.03994) |     [Rice2020Overfitting](https://arxiv.org/abs/2002.11569) |     **Remark**     |
| :----------------: | :-----------------: | -------------------------------------------: | -------------------------------------------: | ---------------------------------------------: | :----------------: |
|      **FGSM** (Linf)      |    Torchattacks     | 34% (54ms) |                                 **48% (5ms)** |                                    62% (82ms) |                    |
|  | **Foolbox<sup>*</sup>** | **34% (15ms)** |                                     48% (8ms) |                  **62% (30ms)** |                    |
|                    |         ART         | 34% (214ms) |                                     48% (59ms) |                                   62% (768ms) |                    |
| **PGD** (Linf) |    **Torchattacks** | **0% (174ms)** |                               **44% (52ms)** |            **58% (1348ms)** | :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% (354ms) |                                  44% (56ms) |              58% (1856ms) |                    |
|                    |         ART         | 0% (1384 ms) |                                   44% (437ms) |                58% (4704ms) |                    |
| **CW<sup>†?</sup>**(L2) |    **Torchattacks** | **0% / 0.40<br /> (2596ms)** |                **14% / 0.61 <br />(3795ms)** | **22% / 0.56<br />(43484ms)** | :crown: **Highest Success Rate** <br /> :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.40<br /> (2668ms) |                   32% / 0.41 <br />(3928ms) |                34% / 0.43<br />(44418ms) |  |
|                    |         ART         | 0% / 0.59<br /> (196738ms) |                 24% / 0.70 <br />(66067ms) | 26% / 0.65<br />(694972ms) |  |
| **PGD** (L2) |    **Torchattacks** | **0% / 0.41 (184ms)** |                  **68% / 0.5<br /> (52ms)** |                  **70% / 0.5<br />(1377ms)** | :crown: **Fastest** |
|                    | Foolbox<sup>*</sup> | 0% / 0.41 (396ms) |                       68% / 0.5<br /> (57ms) |                     70% / 0.5<br /> (1968ms) |                    |
|                    |         ART         | 0% / 0.40 (1364ms) |                       68% / 0.5<br /> (429ms) | 70% / 0.5<br /> (4777ms) |                           |

<sup>*</sup> FoolBox는 정확도와 adversarial image를 동시에 반환하기 때문에 실제 image generation 시간은 기재된 것보다 짧을 수 있습니다.


## Contribution
### :star2: Contribution 은 언제나 환영입니다. 
Adversarial Attack은 앞으로도 계속 새롭게 나올 것입니다. YourBench는 추후에도 model의 adversarial robustness를 증명할 때  사용되는 라이브러리가 될 수 있도록 노력하고자 합니다. 앞으로 새로운 adversarial attack이 나온다면 알려주세요! YourBench 에 contribute 하고 싶다면 아래를 참고해주세요.
[CONTRIBUTING.md](CONTRIBUTING.md).



## reference

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
  
