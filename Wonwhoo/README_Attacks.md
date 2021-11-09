# -*- coding: utf-8 -*-
# YourBench - attacks

<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

YourBench�� ������� �𵨰� �Ķ����, �����ͼ��� �Է¹޾� �� 4����(FGSM, CW, PGD, DeepFool)�� ������ �����մϴ�.

## ����

1. [�Է�](#�Է�)
2. [����](#����)
3. [���](#���)
4. [�������](#�������)


```shell
python main.py --pth "pth_���" --model "model_����_���" --dataset "������_���丮" --dataindex "������_�ε���_���丮" --attack_medthod CW FGSM
```

## �Է�

### :point_right: ��� �Է¹ޱ�
YourBench�� ������� �� ���Ǹ� .py�� ���·� �޽��ϴ�. state_dict ������ ����ִ� .pth �Ǵ� .pt�� �Բ� �Է����ּ���. \
����ڷκ��� worst case, average case, best case�� �ش��ϴ� �ִ� 3���� state_dict�� ���� �� �ֽ��ϴ�. 

```python
#�� �ҷ�����
model = "/home/auspiciouswho47/adversarial-attacks-pytorch/demos/lenet_state_dict.pth"
from LeNet_model_definition import Net
model = Net().to(device)
model.eval()
```

### :point_right: �����ͼ� �Է¹ޱ�
������� �����ͼ��� custom data set�̶�� �����ͼ��� �Է��� ���� ���ּ���.
�������� �ε����� ����ִ� json ���ϰ� ���� ������ �ʿ��մϴ�.
```
1.jpg //�׽�Ʈ ������
image_class_index.json//�̹����� �ε����� ���ǵǾ��ִ� json����
```
```python
##json���� ����##
{"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], "3": ["n01491361", "tiger_shark"], "4": ["n01494475", "hammerhead"], "5": ["n01496331", "electric_ray"], "6": ["n01498041", "stingray"], "7": ["n01514668", "cock"], "8": ["n01514859", "hen"], "9": ["n01518878", "ostrich"], "10": ["n01530575", "brambling"], "11": ["n01531178", "goldfinch"], "12": ["n01532829", "house_finch"], "13": ["n01534433", "junco"], "14": ["n01537544", "indigo_bunting"], "15": ["n01558993", "robin"], "16": ["n01560419", "bulbul"], "17": ["n01580077", "jay"], "18": ["n01582220", "magpie"], "19": ["n01592084", "chickadee"], "20": ["n01601694", "water_ouzel"], "21": ["n01608432", "kite"], "22": ["n01614925", "bald_eagle"], "23": ["n01616318", "vulture"], "24": ["n01622779", "great_grey_owl"], "25": ["n01629819", "European_fire_salamander"], "26": ["n01630670", "common_newt"], "27": ["n01631663", "eft"]
```

###  :warning: �������
YourBench�� ���� ��Ȯ�� test�� �����ϰ� report�� �����ϱ� ���ؼ� ���� ������ �𵨿� ���ؼ� ��������� �Ӵϴ�.
* **No Zero Gradients** \
Obfuscated Gradients�� �˷��� Vanishing/Exploding gradients, Shattered Gradients, �׸��� Stochastic Gradients�� ����ϴ� �𵨿� ���ؼ��� ����� �������� �ʽ��ϴ�. �� gradients�� ����ϴ� ���� ������ ��� ����� �ƴϸ�, adversarial attack generation�� �ſ� ����ϴ�. Obfuscated gradients�� ����ϴ� �𵨵��� EOT�� BPDA, Reparameterizing�� ���� �����ϴ� ���� �����մϴ�.
* **No Loops in Forward Pass** \
Forward pass�� loop�� �ִ� ���� backpropagation�� ����� ������Ű��, �ð��� ���� �ɸ��� �մϴ�. �̷��� �𵨵鿡 ���ؼ� loop�� loss�� �ش� ���� task�� ���Ͽ� ���������� ������ �� �ִ� ������ �����մϴ�.

## ����
YourBench�� 4���� ������ �����մϴ�.\
������ �䱸�ϴ� �Ķ������ �⺻���� ��� ���� �����մϴ�.

### :sunny: Vanilla
input image�� �״�� �����մϴ�.\
�Ķ���ͷ� model���� �޽��ϴ�.
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
* ��뿹��
```python
attack = yourbench.VANILLA(model)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: FGSM
 **��Explaining and harnessing adversarial examples�� [https://arxiv.org/abs/1412.6572]**\
 FGSM�� Linf norm�� ����ϴ� �����Դϴ�.\
�Ķ���ͷ� model�� eps�� �޽��ϴ�.\
eps(float): �ִ� ���� (maximum perturbation) (Default: 0.007)
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
* ��뿹��
```python
attack = torchattacks.FGSM(model, eps=0.007)
adv_images = attack(images, labels)
```

### :cloud_with_lightning_and_rain: CW
**��Towards Evaluating the Robustness of Neural Networks�� [https://arxiv.org/abs/1608.04644]**\
**CW�� L2 norm�� ����ϴ� �����Դϴ�.**\
**�Ķ���ͷ� model, c, kappa, steps, lr�� �޽��ϴ�.**\
**c(float) : box-constraint�� ���� ���Դϴ�. (Default: 1e-4)**\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140865502-7f6c076f-d0a9-45c8-b585-88d0af81c6e4.png)
\
**kappa(float) : ������ confidence�� �����մϴ�. (Default: 0)**\
$f(x��)=max(max{Z(x��)i:i��t}?Z(x��)t,?��)$
**steps (int) : ������ �ܰ� (Default: 1000)**\
**lr (float) : Adam optimizer�� learning rate (Default: 0.01)**

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
* ��뿹��
```python
attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: PGD
��Towards Deep Learning Models Resistant to Adversarial Attacks�� [https://arxiv.org/abs/1706.06083]\
�Ķ���ͷ� model, eps, alpha, steps, random_start�� �޽��ϴ�.\
model (nn.Module) : ������ ��.\
eps (float) : �ִ� ���� (maximum perturbation). (Default: 0.3)\
alpha (float) : step�� ũ�� (Default: 2/255)\
steps (int) : step Ƚ��. (Default: 40)\
random_start (bool) : delta�� ���� �ʱ�ȭ ����. (Default: True)\
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
* ��뿹��
```python
attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
DeepFool�� L2 norm�� ����ϴ� �����Դϴ�.\
�Ķ���ͷ� model, steps, overshoot�� �޽��ϴ�.\
model (nn.Module) : ������ ��\
steps (int) : step�� ����. (Default: 50)\
overshoot (float) : noise ������ ���� �Ķ����. (Default: 0.02)
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
* ��뿹��
```python
attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
adv_images = attack(images, labels)
```

## ���
���� ���丮�� report.pdf������ �����˴ϴ�. \
pdf�� ���ô� �Ʒ��� �����ϴ�.
### accuracy against attacks
���� ���� ����� ��ǥ�� ���� �����ݴϴ�. \
���� ������� model�� accuracy (robustness)�� ������ �����ϴµ� �ɸ� total elapsed time�� ����Ǿ��ֽ��ϴ�.
### attack results with graph
ǥ�� ���� ����� �׷����� ��ȯ�Ͽ� �����մϴ�.
### adversarial examples
������ adversarial examples �� ���ݿ� ������ �̹����� �����ݴϴ�.\
�̋� ���� � ���̺�� �ν��ߴ��� ���� ���� �����ݴϴ�.
### advises
���� �� robust�� ���� ���� �����ڰ� � ��ġ�� ���ؾ��ϴ��� ���̵������ �����ֽ��ϴ�.
</br>\
![adversarial attack-17](https://user-images.githubusercontent.com/80820556/136702057-26e82c95-8536-4619-b9b1-d8cda12d9c55.jpg)

</p></details>


##  �������
������ ���� ����� ��Ŀ���� ���� �ñ��ϴٸ� �Ʒ� ������ ������ ����ٶ��ϴ�.
### :fairy: FGSM
��Explaining and harnessing adversarial examples�� [https://arxiv.org/abs/1412.6572]\
Fast Gradient Signed Method, FGSM�� Ian Goodfellow et al. �� ������ adversarial attack�Դϴ�.\
���� �ſ� ���� ���, �з���� x�� x'�� ���� class�� �����մϴ�.\
���� ������ ����� ������ �����ϴ�.\
\
![gif](https://user-images.githubusercontent.com/80820556/140694562-c7fcf1f1-85f4-431b-b6f8-ce002ffefd74.gif)\
$w :$ weight vector\
$x':$ adversarial example\
$x :$ input to the model\
$�� :$ perturbation\
\
�̶� max norm contraint�� ���� ��=sign(w)�� �� perturbation�� �ִ�ȭ ��ų �� �ֽ��ϴ�.\
$�� = ��*sign(w) = ��||w||$\
w�� n������ �����̰�, element�� ���� ����� m�̶�� �簪�� ��mn�� �˴ϴ�.\
w��� ����n�� ����ϱ� ������ �� ������, ���� ������ �������� input�� ���� ���̰� output�� ū ���̸� ���� �� �ս��ϴ�.\
��, ���� �������� input�� ���� ����� �߰��Ͽ� Decision Boundary�� ũ�� �ѱ� �� �ֽ��ϴ�.\
�������� input�� ����� ������ �ִ� ���, ������ ���� �𵨿� adversarial example�� �ִٰ� �Ͻ��մϴ�.\
\
FGSM�� ��ǥ���� one-step ���� �˰����Դϴ�.\
���� ���ĸ� (steepest) �������� optimization loss J(��, ?, ?)
�� ������Ű�� ���� loss�� gradient ���� �� ���� �̹����� �����մϴ�.\
������ ���� x���� ������ ���� �����˴ϴ�.\
$x'= x - \epsilon *sign(\Delta_xJ(\Theta , x, y))$\
$x :$ input to the model\
$x':$ adversarial example\
$J:$ optimization loss (adversarial loss)\
$��: $ perturbation\
$x: $ adversarial example\
$y':$ iamge label\
loss�� �ش�ȭ���� ���з��� �����ؾ��ϱ� ������ loss�� �����մϴ�.\
gradient�� backpropagation���� ����� �� �ֽ��ϴ�.\

### :fairy: CW
��Towards Evaluating the Robustness of Neural Networks�� [https://arxiv.org/abs/1608.04644]\
Carlini�� Wagner�� L0, L2, L�ļ� ���� metric�� �̿��Ͽ� ������ ������ �����س��� ����ȭ ����� ������ ������ �����Ͽ����ϴ�.\
����ȭ �����Լ��� ������ �����ϴ�.\
$min_�� D(x, x+��) +c * f(x+��)$\
$��:$ perturbation\
$D(��, ��):$ distance metric\
$f(��):$ loss term (classifier�� �Է��� Ʋ���� �з��Ͽ��� �� 0������ ���� ��ȯ)\
\
f(?��)�� ������ ���� ���ǵ˴ϴ�.\
$f(x') = max(Z(x')_{l_x} - max Z(x')_i : i��l_x, -K)$\
$Z(?��):$ classifier�� logit(softmax �Լ� ���� �� vector)\
$?:$ confidence\
CW attack�� �Ķ���� ������ �����Ͽ� ������ ������ ���� �� �� �ִٴ� ������ �ֽ��ϴ�.\
?�� ���� Ŭ���� classifier�� ������ ������ �� ���� confidence�� Ʋ���� �з��մϴ�.\

### :fairy: PGD
��Towards Deep Learning Models Resistant to Adversarial Attacks�� [https://arxiv.org/abs/1706.06083]\
Projected Gradient Descent ������ FGSM�� ���� ���� �� 3�� �ڿ� ���� ���� ����Դϴ�.\
��������� universial first-order adversary�� �˷��� �־� ���� ������ baseline ���ݹ������ ����˴ϴ�.\
FGSM�� ������ ������� n���� step��ŭ ������ �ݺ��Ͽ� ������ $L_{infinity}$ norm �Ʒ����� inner maximization�� �����մϴ�.\
$X_0 = X, X_{i+1} = ��(X_i + ��*sign(��_xL(X_t, y_{true})))$\
\
�������� PGD����� ������ ���� ã�Ƴ� local maxima�� ��, �����ͼ¿� ������� ����� �սǰ����� �����ϴ� ���� ���������� �����߽��ϴ�.\
�� ����� �������� ���� ���з��� �����ϱ� ���� local maxima�� ã�� �����ظ� ���ϱ� ���� first-order���� ����� ���� �߿��� PGD�� ����ϴ� ���� ���� ȿ�����̶�� �����մϴ�.\
������ ���� ������ PDG example�� �Ʒý�Ų adversarial trained ���� ��� ���ݿ��� �ϰ��� ������ �����ݴϴ�.
\
FGSM������ optimal �並 ã�� ���� 1 step gradient�� ����մϴ�.\
PGD�� ��� step�� n�� ���� ���� ������ �������ϴ�.\
�Ϲ������� 7, 40���� ����ϰ� (Default) ������� ������ ���� �� ������ local optima�� ã�� ���� step���� ������ �� �ֽ��ϴ�.\
������ ������ �ս��Լ� ���� Ư�� ���� ������ �����ϴ� ������ ��Ÿ���ϴ�.\
��, step���� Ŀ�� ���� �� ����� ������ �����ϱ� ������ ������ step���� �����ϴ� ���� �߿��մϴ�.\

### :fairy: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
Moosavi-Dezfooli ���� ������ DeepFool �� Ÿ�� ���� �����̶�� �����ϰ� ������ ���� ?�� �� ã���ϴ�.\
Input image ?�� ���� ����� decision boundary�� ã��, �� �������� ?�Ǹ� �����մϴ�.\
?�ǰ� decision boundary�� �Ѿ ������ �ش� ������ �ݺ��ϰ�, ���� ũ���� perturbation���� ������ ������ ã�� �� �ֽ���
��.


