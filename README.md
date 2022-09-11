### Mlp-mixer

Implementation for paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) with [SERF: Towards better training of deep neural networks using log-Softplus ERror activation Function](https://arxiv.org/abs/2108.09598)

#### Serf function

$f(x)=x*erf(ln(1+e^x))$

#### Mlp-mixer configs

<img src="https://github.com/bdghuy/Mlp-mixer/blob/main/configs.PNG" width="433" height="120">

#### Sample Usage

```
mixer_S16 = MLPMixer(input_shape=(im_height,im_width,3),
                     num_classes = num_classes,
                     N = 8,
                     P = 16,
                     C = 512,
                     DS = 256,
                     DC = 2048,
                     dropout_rate = 0.2)
```
