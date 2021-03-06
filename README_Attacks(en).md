# YourBench - attacks

<p>
  <a href="https://github.com/jonye/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/jonyejin/YourBench?&color=brightgreen" /></a>

YourBench performs four attacks (FGSM, CW, PGD, DeepFool) by receiving the user's model, parameters, and dataset as input.

## 목차

1. [Input](#Input)
2. [Attack](#Attack)
3. [Result](#Result)
4. [Reference](#Reference)


```shell
python main.py --pth "pth_route" --model "model_definition_route" --dataset "data_directory" --dataindex "data_index_directory" --attack_medthod CW FGSM
```

## Input

### :point_right: Get module input
YourBench receives your model definition in the form of a .py. Please enter it with .pth or .pt containing state_dict information. \
You can receive up to 3 state_dicts from the user, corresponding to the worst case, average case, and best case.

```python
#Import model
model = "/home/auspiciouswho47/adversarial-attacks-pytorch/demos/lenet_state_dict.pth"
from LeNet_model_definition import Net
model = Net().to(device)
model.eval()
```

### :point_right: Get data set input
If your data set is a custom data set, enter the data set separately.
Need a json file containing an index of the data and a picture file.
```
1.jpg //test data
image_class_index.json// json file where the index of the image is defined
```
```python
##json file example##
{"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "2": ["n01484850", "great_white_shark"], "3": ["n01491361", "tiger_shark"], "4": ["n01494475", "hammerhead"], "5": ["n01496331", "electric_ray"], "6": ["n01498041", "stingray"], "7": ["n01514668", "cock"], "8": ["n01514859", "hen"], "9": ["n01518878", "ostrich"], "10": ["n01530575", "brambling"], "11": ["n01531178", "goldfinch"], "12": ["n01532829", "house_finch"], "13": ["n01534433", "junco"], "14": ["n01537544", "indigo_bunting"], "15": ["n01558993", "robin"], "16": ["n01560419", "bulbul"], "17": ["n01580077", "jay"], "18": ["n01582220", "magpie"], "19": ["n01592084", "chickadee"], "20": ["n01601694", "water_ouzel"], "21": ["n01608432", "kite"], "22": ["n01614925", "bald_eagle"], "23": ["n01616318", "vulture"], "24": ["n01622779", "great_grey_owl"], "25": ["n01629819", "European_fire_salamander"], "26": ["n01630670", "common_newt"], "27": ["n01631663", "eft"]
```

###  :warning: Constraint
YourBench places constraints on the measurable model to perform more accurate tests and provide reports.
* **No Zero Gradients** \
Not recommended for models that use Vanishing/Exploding gradients, Shattered Gradients, and Stochastic Gradients, also known as Obfuscated Gradients. The model using the above gradients is not a suitable defense technique, and adversarial attack generation is very difficult. Models using obfuscated gradients are recommended to attack through EOT, BPDA, and reparameterizing.
* **No Loops in Forward Pass** \
Models with loops in the forward pass increase the cost of backpropagation and take a long time. For these models, we recommend an attack that can be adaptively applied by combining the loop loss and the task of the model.
## Attack
YourBench offers 4 attacks.\
The default values of the parameters required by the attack are all referenced in the paper.

### :sunny: Vanilla
Returns the input image as is.\
It takes only model as parameter.
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
* Example of use
```python
attack = yourbench.VANILLA(model)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: FGSM
 **‘Explaining and harnessing adversarial examples’ [https://arxiv.org/abs/1412.6572]**\
 FGSM is an attack using the Linf norm.\
It takes model and eps as parameters.\
eps(float): maximum perturbation (Default: 0.007)
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
* Example of use
```python
attack = torchattacks.FGSM(model, eps=0.007)
adv_images = attack(images, labels)
```

### :cloud_with_lightning_and_rain: CW
**‘Towards Evaluating the Robustness of Neural Networks’ [https://arxiv.org/abs/1608.04644]**\
**CW is an attack using the L2 norm.**\
**It takes model, c, kappa, steps and lr as parameters.**\
**c(float) : The value for the box-constraint. (Default: 1e-4)**\
![lagrida_latex_editor (2)](https://user-images.githubusercontent.com/80820556/140870613-5f61196a-54d6-4220-87f5-09a1188b0a9e.png)\
**kappa(float) : Appears with confidence in the paper. (Default: 0)**\
![lagrida_latex_editor (3)](https://user-images.githubusercontent.com/80820556/140870614-967c2adf-d54d-4c85-9271-0162355607f9.png)
\
**steps (int) : step in progress (Default: 1000)**\
**lr (float) : Adam optimizer's learning rate (Default: 0.01)**

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
* Example of use
```python
attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: PGD
‘Towards Deep Learning Models Resistant to Adversarial Attacks’ [https://arxiv.org/abs/1706.06083]\
It takes model, eps, alpha, steps, random_start as parameters.\
model (nn.Module) : model to attack.\
eps (float) : maximum perturbation. (Default: 0.3)\
alpha (float) : size of step (Default: 2/255)\
steps (int) : number of steps. (Default: 40)\
random_start (bool) : Whether delta is initialized at random. (Default: True)\
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
* Example of use
```python
attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
adv_images = attack(images, labels)
```
### :cloud_with_lightning_and_rain: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
DeepFool is an attack that uses the L2 norm.\
It takes model, steps and overshoot as parameters.\
model (nn.Module) : model to attack\
steps (int) : number of steps. (Default: 50)\
overshoot (float) : Parameters for noise amplification. (Default: 0.02)
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
* Example of use
```python
attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
adv_images = attack(images, labels)
```

## Result
Report.pdf file is created in the save directory. \
Example of a pdf is below.
### accuracy against attacks
Shows the attack performance results in a diagram. \
In the execution result, the accuracy (robustness) of the model and the total elapsed time taken to execute the attack are described.
### attack results with graph
The table results are converted into graphs and presented.
### adversarial examples
Among the generated adversarial examples, it shows an image that successfully attacked.\
It also shows what label this model recognized.
### advises
There are guidelines on what actions developers should take for more robust models.
</br>\
![adversarial attack-17](https://user-images.githubusercontent.com/80820556/136702057-26e82c95-8536-4619-b9b1-d8cda12d9c55.jpg)

</p></details>


##  Reference
If you are curious about the details and mechanism of the attack, please refer to the original text and contents below.
### :fairy: FGSM
‘Explaining and harnessing adversarial examples’ [https://arxiv.org/abs/1412.6572]\
Fast Gradient, Signed Method and FGSM are described by Ian Goodfellow et al. This is an adversarial attack suggested by this.\
If η is very small, the classifier classifies x and x' into the same class.\
The relationship between the values is like this.\
\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140866522-4ceeab47-06d6-4824-a948-42b15e958613.png)
\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140866950-340b12eb-8804-4051-a545-c8bc8f8408b2.png)
\
![lagrida_latex_editor (2)](https://user-images.githubusercontent.com/80820556/140866808-a656465b-c141-4457-ba0d-7cf5848c2990.png)
\
![lagrida_latex_editor (3)](https://user-images.githubusercontent.com/80820556/140866854-6c3b7d90-d466-4264-a202-c8da305b4e90.png)
\
![lagrida_latex_editor (4)](https://user-images.githubusercontent.com/80820556/140866887-dd166c33-3dc0-4f54-93ac-7bb3554ae796.png)
\
\
At this time, this perturbation can be maximized with η=sign(w) according to the max norm constraint.\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140867054-cd759d0b-135a-4022-8d80-61b7224b7cdd.png)
\
If w is an n-dimensional vector and the mean of the absolute values of the elements is m, then the value of η is εmn.\
wη can increase proportionally to dimension n, and in high-dimensional problems, small differences in input can make large differences in output.\
In other words, at a high dimensionality, by adding a small noise to the input, you can significantly bypass the Decision Boundary.\
The paper suggests that a simple linear model has an adversarial example if the input has enough dimensions.\
\
FGSM is a representative one-step attack algorithm.\
Update the image along the gradient direction of the loss to increase J(θ, ?, ?) in the steepest direction.


The adversarial example x′ is generated like this\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140867337-93debfbf-4ae0-4399-8c35-b2d2ae43c206.png)\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140867335-f40cb6e3-a652-46ef-a846-6674c7bc4325.png)\
![lagrida_latex_editor (2)](https://user-images.githubusercontent.com/80820556/140867333-3b2015a3-b2d8-4bfd-8194-5b16c095016e.png)\
![lagrida_latex_editor (3)](https://user-images.githubusercontent.com/80820556/140867332-352b8685-5cb7-405e-92bd-0bf3cde3a1e1.png)
\
![lagrida_latex_editor (4)](https://user-images.githubusercontent.com/80820556/140867343-e93515f6-77be-44ed-85b5-deda44e14877.png)\
![lagrida_latex_editor (5)](https://user-images.githubusercontent.com/80820556/140867340-1f2d6ee6-4be0-4fe2-aa94-c51f9663aeab.png)\
![lagrida_latex_editor (6)](https://user-images.githubusercontent.com/80820556/140867339-223b588e-dc7f-49ed-bb19-30c53a05eb75.png)\
We subtract the loss because we need to maximize the loss to induce misclassification.\
gradient can be computed with backpropagation.

### :fairy: CW
‘Towards Evaluating the Robustness of Neural Networks’ [https://arxiv.org/abs/1608.04644]\
Carlini and Wagner proposed an optimization-based adversarial attack that generates adversarial examples using three metrics L0, L2, and L∞.\
The optimization objective function like this.\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140867829-12d01026-04a8-4d65-93be-251d46563175.png)\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140867828-bd1bb87a-39a0-409e-a0e2-51e55296167a.png)\
![lagrida_latex_editor (2)](https://user-images.githubusercontent.com/80820556/140867824-470ea915-0e98-49c3-8a88-7f5e047282a7.png)\
![lagrida_latex_editor (3)](https://user-images.githubusercontent.com/80820556/140867832-c7f56f0a-1fe3-44d5-b95f-251cbf427dfa.png)\
\
f(x') is defined as\
![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140869591-9d21badd-3c47-458e-ac54-c6ec9c34bdb2.png)\
![lagrida_latex_editor (5)](https://user-images.githubusercontent.com/80820556/140867830-7ba8762f-21f7-46dc-9f07-4bd48e9f2ab7.png)\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140869637-ce2858b7-db1a-4546-8c04-173afbf2ce57.png)
\
CW attack has the advantage of being able to control the strength of the attack by adjusting the parameter values.\
Larger values of ? cause the classifier to incorrectly classify adversarial examples with higher confidence.

### :fairy: PGD
‘Towards Deep Learning Models Resistant to Adversarial Attacks’ [https://arxiv.org/abs/1706.06083]\
Projected Gradient Descent attack is an attack method that came out about three years after the advent of FGSM.\
Even today, it is known as a universal first-order adversary and is used as a baseline attack method in many papers.\
In a method applying FGSM, the attack is repeated as many as n steps to perform inner maximization under the determined  ![lagrida_latex_editor](https://user-images.githubusercontent.com/80820556/140869893-19d6de49-dd3e-41b0-90d0-e2db3e3f95b8.png) norm.\
![lagrida_latex_editor (1)](https://user-images.githubusercontent.com/80820556/140869899-c2590a17-944c-42e6-9cb9-6a815e03a14c.png)\
\
In this paper, it was experimentally proven that the local maxima found through the PGD-based attack converge to a similar loss value regardless of the model or dataset.\
Based on this fact, it is argued that using PGD is the most effective among first-order attacks to find the optimal solution to find the local maxima for inducing misclassification of the model.\
In fact, the adversarial trained model trained on PDG examples in several papers shows consistent performance in any attack.
\
In FGSM, one step gradient is calculated to find the optimal δ.\
In the case of PGD, the attack strength increases according to the step n.\
In general, 7, 40, etc. are used (Default), and the number of steps can be increased to find a more sophisticated local optima according to the user's adjustment.\
However, in the paper, it appears that the loss function value converges rapidly to a specific value.\
In other words, as the number of steps increases, the degree of influence decreases, so it is important to adjust the number of steps appropriately.\

### :fairy: DeepFool
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks' [https://arxiv.org/abs/1511.04599]\
DeepFool as suggested by Moosavi-Dezfooli et al. assumes that the target model is linear and finds the adversarial example ?'.\
Find the decision boundary closest to the input image ?, and update ?' in this direction.\
We repeat the process until ?′ crosses the decision boundary, and find adversarial examples with small-scale perturbations.
