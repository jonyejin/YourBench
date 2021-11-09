# -*- coding: utf-8 -*-
# YourBench - attacks

<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

YourBench는 사용자의 모델과 파라미터, 데이터셋을 입력받아 총 4가지(FGSM, CW, PGD, DeepFool)의 공격을 수행합니다.

## 목차

1. [입력](#입력)
2. [공격](#공격)
3. [결과](#결과)
4. [참고사항](#참고사항)


```shell
python main.py --pth "pth_경로" --model "model_정의_경로" --dataset "데이터_디렉토리" --dataindex "데이터_인덱스_디렉토리" --attack_medthod CW FGSM
```

## 입력

### :point_right: 모듈 입력받기
YourBench는 사용자의 모델 정의를 .py의 형태로 받습니다. state_dict 정보가 담겨있는 .pth 또는 .pt와 함께 입력해주세요. \
사용자로부터 worst case, average case, best case에 해당하는 최대 3개의 state_dict를 받을 수 있습니다. 

```python
#모델 불러오기
model = "/home/auspiciouswho47/adversarial-attacks-pytorch/demos/lenet_state_dict.pth"
from LeNet_model_definition import Net
model = Net().to(device)
model.eval()
```

### :point_right: 데이터셋 입력받기
사용자의 데이터셋이 custom data set이라면 데이터셋을 입력을 따로 해주세요.
데이터의 인덱스가 들어있는 json 파일과 사진 파일이 필요합니다.
```
1.jpg //테스트 데이터
image_class_index.json//이미지의 인덱스가 정의되어있는 json파일
```
```python
##json파일 예시##
{"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], "3": ["n01491361", "tiger_shark"], "4": ["n01494475", "hammerhead"], "5": ["n01496331", "electric_ray"], "6": ["n01498041", "stingray"], "7": ["n01514668", "cock"], "8": ["n01514859", "hen"], "9": ["n01518878", "ostrich"], "10": ["n01530575", "brambling"], "11": ["n01531178", "goldfinch"], "12": ["n01532829", "house_finch"], "13": ["n01534433", "junco"], "14": ["n01537544", "indigo_bunting"], "15": ["n01558993", "robin"], "16": ["n01560419", "bulbul"], "17": ["n01580077", "jay"], "18": ["n01582220", "magpie"], "19": ["n01592084", "chickadee"], "20": ["n01601694", "water_ouzel"], "21": ["n01608432", "kite"], "22": ["n01614925", "bald_eagle"], "23": ["n01616318", "vulture"], "24": ["n01622779", "great_grey_owl"], "25": ["n01629819", "European_fire_salamander"], "26": ["n01630670", "common_newt"], "27": ["n01631663", "eft"]
```

###  :warning: 제약사항
YourBench는 보다 정확한 test를 수행하고 report를 제공하기 위해서 측정 가능한 모델에 대해서 제약사항을 둡니다.
* **No Zero Gradients** \
Obfuscated Gradients로 알려진 Vanishing/Exploding gradients, Shattered Gradients, 그리고 Stochastic Gradients를 사용하는 모델에 대해서는 사용을 권장하지 않습니다. 위 gradients를 사용하는 모델은 적합한 방어 기법이 아니며, adversarial attack generation이 매우 힘듭니다. Obfuscated gradients를 사용하는 모델들은 EOT나 BPDA, Reparameterizing을 통해 공격하는 것을 권장합니다.
* **No Loops in Forward Pass** \
Forward pass에 loop가 있는 모델은 backpropagation의 비용을 증가시키고, 시간이 오래 걸리게 합니다. 이러한 모델들에 대해선 loop의 loss와 해당 모델의 task를 합하여 적응적으로 적용할 수 있는 공격을 권장합니다.

## 공격
YourBench는 4가지 공격을 제공합니다.\
공격이 요구하는 파라미터의 기본값은 모두 논문을 참조합니다.

### :sunny: Vanilla
input image를 그대로 리턴합니다.\
파라미터로 model만을 받습니다.
```python
class VANILA(Attack):
    """
    Vanila version of Attack.
    It just returns the input images.

    Arguments:
        model (nn.Module): model to attack.

    Shape:
        - images: :math:`(N, C, H, W)`
          where `N = number of batches`, `C = number of channels`,        
                `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VANILA(model)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model):
        super().__init__("VANILA", model)
        self._supported_mode = ['default']

    def forward(self, images, labels=None):
        """
        Overridden.
        """
        adv_images = images.clone().detach().to(self.device)

        return adv_images
```
* 사용예시
```python
attack = yourbench.VANILLA(model)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: FGSM
 **‘Explaining and harnessing adversarial examples’ [https://arxiv.org/abs/1412.6572]**\
 FGSM은 Linf norm을 사용하는 공격입니다.\
파라미터로 model과 eps를 받습니다.\
eps(float): 최대 섭동 (maximum perturbation) (Default: 0.007)
```python
class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)`
          where `N = number of batches`, `C = number of channels`, 
                `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
```
* 사용예시
```python
attack = torchattacks.FGSM(model, eps=0.007)
adv_images = attack(images, labels)
```

### :cloud_with_lightning_and_rain: CW
**‘Towards Evaluating the Robustness of Neural Networks’ [https://arxiv.org/abs/1608.04644]**\
**CW는 L2 norm을 사용하는 공격입니다.**\
**파라미터로 model, c, kappa, steps, lr을 받습니다.**\
**c(float) : box-constraint를 위한 값입니다. (Default: 1e-4)**\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140865502-7f6c076f-d0a9-45c8-b585-88d0af81c6e4.png)
\
**kappa(float) : 논문에서 confidence로 등장합니다. (Default: 0)**\
$f(x′)=max(max{Z(x′)i:i≠t}?Z(x′)t,?κ)$
**steps (int) : 진행할 단계 (Default: 1000)**\
**lr (float) : Adam optimizer의 learning rate (Default: 0.01)**

```python
class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` 
          where `N = number of batches`, `C = number of channels`,   
                `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            if step % (self.steps//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images


    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)
```
* 사용예시
```python
attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: PGD
‘Towards Deep Learning Models Resistant to Adversarial Attacks’ [https://arxiv.org/abs/1706.06083]\
파라미터로 model, eps, alpha, steps, random_start를 받습니다.\
model (nn.Module) : 공격할 모델.\
eps (float) : 최대 섭동 (maximum perturbation). (Default: 0.3)\
alpha (float) : step의 크기 (Default: 2/255)\
steps (int) : step 횟수. (Default: 40)\
random_start (bool) : delta의 랜덤 초기화 여부. (Default: True)\
```python
class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` 
                   where `N = number of batches`, `C = number of channels`,   
                         `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
```
* 사용예시
```python
attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
DeepFool은 L2 norm을 사용하는 공격입니다.\
파라미터로 model, steps, overshoot를 받습니다.\
model (nn.Module) : 공격할 모델\
steps (int) : step의 갯수. (Default: 50)\
overshoot (float) : noise 증폭을 위한 파라미터. (Default: 0.02)
```python
class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)`
                   where `N = number of batches`, `C = number of channels`,   
                         `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images


    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
```
* 사용예시
```python
attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
adv_images = attack(images, labels)
```

## 결과
저장 디렉토리에 report.pdf파일이 생성됩니다. \
pdf의 예시는 아래와 같습니다.
### accuracy against attacks
공격 수행 결과를 도표를 통해 보여줍니다. \
수행 결과에는 model의 accuracy (robustness)와 공격을 수행하는데 걸린 total elapsed time이 기재되어있습니다.
### attack results with graph
표로 나온 결과를 그래프로 변환하여 제시합니다.
### adversarial examples
생성된 adversarial examples 중 공격에 성공한 이미지를 보여줍니다.\
이떄 모델이 어떤 레이블로 인식했는지 또한 같이 보여줍니다.
### advises
보다 더 robust한 모델을 위해 개발자가 어떤 조치를 취해야하는지 가이드라인이 적혀있습니다.
</br>\
![adversarial attack-17](https://user-images.githubusercontent.com/80820556/136702057-26e82c95-8536-4619-b9b1-d8cda12d9c55.jpg)

</p></details>


##  참고사항
공격의 세부 내용과 메커니즘에 대해 궁금하다면 아래 원문과 내용을 참고바랍니다.
### :fairy: FGSM
‘Explaining and harnessing adversarial examples’ [https://arxiv.org/abs/1412.6572]\
Fast Gradient Signed Method, FGSM은 Ian Goodfellow et al. 이 제시한 adversarial attack입니다.\
η이 매우 작을 경우, 분류기는 x와 x'을 같은 class로 구분합니다.\
값들 사이의 관계는 다음과 같습니다.\
\
![gif](https://user-images.githubusercontent.com/80820556/140694562-c7fcf1f1-85f4-431b-b6f8-ce002ffefd74.gif)\
$w :$ weight vector\
$x':$ adversarial example\
$x :$ input to the model\
$η :$ perturbation\
\
이때 max norm contraint에 따라 η=sign(w)로 이 perturbation을 최대화 시킬 수 있습니다.\
$η = ε*sign(w) = ε||w||$\
w가 n차원의 벡터이고, element의 절댓값 평균이 m이라면 η값은 εmn이 됩니다.\
wη는 차원n에 비례하기 증가할 수 있으며, 높은 차원의 문제에서 input에 작은 차이가 output에 큰 차이를 만들 수 잇습니다.\
즉, 높은 차원에서 input에 작은 노이즈를 추가하여 Decision Boundary를 크게 넘길 수 있습니다.\
논문에서는 input에 충분한 차원이 있는 경우, 간단한 선형 모델에 adversarial example이 있다고 암시합니다.\
\
FGSM은 대표적인 one-step 공격 알고리즘입니다.\
가장 가파른 (steepest) 방향으로 optimization loss J(θ, ?, ?)
를 증가시키기 위해 loss의 gradient 방향 을 따라 이미지를 갱신합니다.\
적대적 예제 x′은 다음과 같이 생성됩니다.\
$x'= x - \epsilon *sign(\Delta_xJ(\Theta , x, y))$\
$x :$ input to the model\
$x':$ adversarial example\
$J:$ optimization loss (adversarial loss)\
$θ: $ perturbation\
$x: $ adversarial example\
$y':$ iamge label\
loss를 극대화시켜 오분류를 유도해야하기 때문에 loss를 감산합니다.\
gradient는 backpropagation으로 계산할 수 있습니다.\

### :fairy: CW
‘Towards Evaluating the Robustness of Neural Networks’ [https://arxiv.org/abs/1608.04644]\
Carlini와 Wagner은 L0, L2, L∞세 개의 metric을 이용하여 적대적 예제를 생성해내는 최적화 기반의 적대적 공격을 제안하였습니다.\
최적화 목적함수는 다음과 같습니다.\
$min_δ D(x, x+δ) +c * f(x+δ)$\
$δ:$ perturbation\
$D(·, ·):$ distance metric\
$f(·):$ loss term (classifier가 입력을 틀리게 분류하였을 때 0이하의 값을 반환)\
\
f(?’)은 다음과 같이 정의됩니다.\
$f(x') = max(Z(x')_{l_x} - max Z(x')_i : i≠l_x, -K)$\
$Z(?’):$ classifier의 logit(softmax 함수 직전 의 vector)\
$?:$ confidence\
CW attack은 파라미터 값들을 조정하여 공격의 강도를 조절 할 수 있다는 장점이 있습니다.\
?의 값이 클수록 classifier가 적대적 예제를 더 높은 confidence로 틀리게 분류합니다.\

### :fairy: PGD
‘Towards Deep Learning Models Resistant to Adversarial Attacks’ [https://arxiv.org/abs/1706.06083]\
Projected Gradient Descent 공격은 FGSM의 등장 이후 약 3년 뒤에 나온 공격 방법입니다.\
현재까지도 universial first-order adversary로 알려져 있어 많은 논문들의 baseline 공격방법으로 차용됩니다.\
FGSM을 응용한 방법으로 n번의 step만큼 공격을 반복하여 정해진 $L_{infinity}$ norm 아래에서 inner maximization을 수행합니다.\
$X_0 = X, X_{i+1} = Π(X_i + α*sign(∇_xL(X_t, y_{true})))$\
\
논문에서는 PGD기반의 공격을 통해 찾아낸 local maxima는 모델, 데이터셋에 상관없이 비슷한 손실값으로 수렴하는 것을 실험적으로 증명했습니다.\
이 사실을 바탕으로 모델의 오분류를 유도하기 위한 local maxima를 찾는 최적해를 구하기 위해 first-order만을 사용한 공격 중에서 PGD를 사용하는 것이 가장 효과적이라고 주장합니다.\
실제로 여러 논문에서 PDG example을 훈련시킨 adversarial trained 모델은 어떠한 공격에도 일관된 성능을 보여줍니다.
\
FGSM에서는 optimal δ를 찾기 위해 1 step gradient를 계산합니다.\
PGD의 경우 step의 n에 따라 공격 강도가 강해집니다.\
일반적으로 7, 40등을 사용하고 (Default) 사용자의 조정에 따라 더 정교한 local optima를 찾기 위해 step수를 증가할 수 있습니다.\
하지만 논문에서 손실함수 값이 특정 값이 빠르게 수렴하는 것으로 나타납니다.\
즉, step수가 커질 수록 그 영향력 정도가 감소하기 때문에 적절한 step수를 조절하는 것이 중요합니다.\

### :fairy: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
Moosavi-Dezfooli 등이 제안한 DeepFool 은 타겟 모델이 선형이라고 가정하고 적대적 예제 ?′ 을 찾습니다.\
Input image ?와 가장 가까운 decision boundary를 찾고, 이 방향으로 ?′를 갱신합니다.\
?′가 decision boundary를 넘어갈 때까지 해당 과정을 반복하고, 작은 크기의 perturbation으로 적대적 예제를 찾을 수 있습니
다.


