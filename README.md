# YourBench - Pytorch
<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

YourBench는 기존에 있는 모델, 또는 사용자의 모델을 입력받아 adversarial attack 에 얼마나 robust한지 평가를 해주는 pytorch library입니다. 모델 개발자들이 더욱 쉽게 adversarial training을 할 수 있도록 모델의 평가 지표를 Report와 함께 제공합니다.

## 목차

1. [서론](#서론)
2. [사용방법](#사용방법)
3. [성능비교](#성능비교)
4. [Contribution](#contribution)
5. [참고사항](#참고사항)



## 서론

### :mega: 적대적 공격이란?
적대적 공격은 딥러닝 모델을 공격하는 가장 대표적인 방법입니다. 딥러닝 모델을 학습시키는 방법을 역으로 모델 공격에 이용하여 모델이 올바른 예측을 하지 못하도록 방해할 수 있습니다. 인간의 눈에는 똑같은 데이터이지만, 모델에 입력하면 전혀 다른 결과가 나올 수 있는 것이죠. **모델이 테스트 이미지를 잘 분류하더라도, 이러한 적대적 공격에 취약하다면 사용하기 어려울 것입니다.**


### :pencil2: 적대적 학습의 중요성
모델이 test data에 대해서 충분히 신뢰성 있는 결과를 낼지라도, 간단한 데이터 조작에 취약하다면 모델을 쓸 수 없게됩니다. adversarial attack과 model robustness는 경찰과 도둑 관계입니다. 서로 꾸준히 발전하면서 따라잡으려고 하기 때문입니다. 현재 자신의 신경망, 또는 모델이 adversarial attack에 대해서 robust 할지라도, 언제든지 새로운 공격 기법이 나타날 수 있습니다. 따라서 모델 개발자 입장에서 새로운 공격기법에 대해서 늘 대비하는 자세가 중요합니다. **다만 그 비용과 시간이 많이 들기 때문에 자신의 신경망이 현재까지 알려지 있는 강력한 adversarial attack에 얼마나 robust한지 확인하는 프로세스 또한 중요하다고 할 수 있습니다.**

### :bulb: 라이브러리의 목적
다른 라이브러리와는 달리, YourBench는 개인 Neural Network를 입력받아 adversarial attack에 대한 Benchmark 점수를 report와 함께 제공합니다. Report는 모델의 취약점과 그에 대한 개선 방안을 제안합니다.개발자는 이를 통해서 자신의 모델의 안정성이 어느정도 인지 가늠할 수 있을 것입니다. 


## 사용방법
### :hugs: 터미널로 실행하기
```console
usage: main.py [-h] -a [{FGSM,CW,PGD,DeepFool} ...] -m {ResNet101_2,ResNet18,Custom} -d {CIFAR-10,CIFAR-100,ImageNet,Custom}
main.py: error: the following arguments are required: -a/--attack_method, -m/--model, -d/--dataset
```
-a 옵션에 수행할 공격을 입력합니다.\
-m 공격을 수행할 모델을 입력합니다. 사용자 모델 수행하고 싶다면 아래 [custom 모델 불러오기](###:hugs:custom모델불러오기) 설명을 읽어주세요. \
-d 데이터셋을 입력합니다. 사용자 데이터를 입력으로 하고 싶다면 아래 [custom 데이터셋 불러오기](###:hugs:custom데이터셋불러오기) 설명을 읽어주세요.\

### :hugs: custom 모델 불러오기
YourBench에서 제공하는 모델 말고도 사용자 모델을 불러올 수 있습니다.\
YourBench 라이브러리를 받은 후, 사용자 모델이 정의된 파이썬 파일을 custom_net.py로 저장해줍니다.\
custom_net.py안에 자신의 모델 클래스명을 my_model이라고 해주세요.\
모델의 파라미터 값이 저장되어있는 state_dict또한 같은 YourBench 디렉토리 안에 my_model.pth로 저장해주세요.\
이후 터미널에서 -m Custom 옵션을 주면 실행이 됩니다.

```python
elif args.parsedModel == 'Custom':
  #사용자 모델 불러오기
  pkg = __import__('custom_net')
  model_custom = pkg.my_model(pretrained = False)

  #state_dict의 경로를 넣기.
  model_custom.load_state_dict(torch.load('./my_model.pth'))
  
  model = nn.Sequential(
      norm_layer,
      model_custom
  ).cuda()
  
  model = model.eval()
```

### :hugs: custom 데이터셋 불러오기
사용자 데이터 또한 불러올 수 있습니다. 사용자 데이터를 불러오기 위해 아래 2가지를 수행해주세요.
#### 1. 클래스와 인덱스를 정의한 .json파일 저장
Report에 정확한 정보 표현을 하기 위해서 이미지 데이터셋의 클래스 이름과 인덱스를 dictionary 형태로 정의한 .json 파일을 YourBench 디렉토리에 저장해주세요. \
아래는 ImageNet의 class_index.json의 예시입니다.
```python
{"0": ["n01440764", "tench"],
  "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], "3": ["n01491361", "tiger_shark"], "4": ["n01494475", "hammerhead"], "5": ["n01496331", "electric_ray"], "6": ["n01498041", "stingray"], "7": ["n01514668", "cock"], "8": ["n01514859", "hen"], "9": ["n01518878", "ostrich"], "10": ["n01530575", "brambling"], "11": ["n01531178", "goldfinch"], "12": ["n01532829", "house_finch"], "13": ["n01534433", "junco"], "14": ["n01537544", "indigo_bunting"], "15": ["n01558993", "robin"], "16": ["n01560419", "bulbul"], "17": ["n01580077", "jay"], "18": ["n01582220", "magpie"], ... }
```
#### 2. Data 디렉토리 안에 사용자 이미지 저장
Data 디렉토리 안에 'custom_dataset' 디렉토리를 만들어 안에 이미지를 저장해주세요. 아래는 imagenet의 예시입니다.
<img width="443" alt="4" src="https://user-images.githubusercontent.com/80820556/144176524-ea4ce4c4-e2a5-454c-b289-368f13b598f9.png">



###  :warning: 제약사항
YourBench는 보다 정확한 test를 수행하고 report를 제공하기 위해서 측정 가능한 모델에 대해서 제약사항을 둡니다.
* **No Zero Gradients** \
Obfuscated Gradients로 알려진 Vanishing/Exploding gradients, Shattered Gradients, 그리고 Stochastic Gradients를 사용하는 모델에 대해서는 사용을 권장하지 않습니다. 위 gradients를 사용하는 모델은 적합한 방어 기법이 아니며, adversarial attack generation이 매우 힘듭니다. Obfuscated gradients를 사용하는 모델들은 EOT나 BPDA, Reparameterizing을 통해 공격하는 것을 권장합니다.
* **No Loops in Forward Pass** \
Forward pass에 loop가 있는 모델은 backpropagation의 비용을 증가시키고, 시간이 오래 걸리게 합니다. 이러한 모델들에 대해선 loop의 loss와 해당 모델의 task를 합하여 적응적으로 적용할 수 있는 공격을 권장합니다.

### :rocket: 데모

* 내장 모델로 실행하기
```python
import yourbench
atk = yourbench.PGD(model, eps=8/255, alpha=2/255, steps=4)
adv_images = atk(images, labels)
```
* 사용자로부터 데이터셋과 모델, 그리고 이미지 레이블을 입력받아 다음을 수행할 수 있습니다.

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



## 성능비교

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



##  참고사항

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
  
