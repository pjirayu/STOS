import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  #net = models.densenet161()
  net = models.resnet18()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#Flops counter for convolutional networks in pytorch framework:https://github.com/sovrasov/flops-counter.pytorch
'''
Usage tips
This script doesn't take into account torch.nn.functional.* operations. For an instance, if one have a semantic segmentation model and use torch.nn.functional.interpolate to upscale features, these operations won't contribute to overall amount of flops. To avoid that one can use torch.nn.Upsample instead of torch.nn.functional.interpolate.
ptflops launches a given model on a random tensor and estimates amount of computations during inference. Complicated models can have several inputs, some of them could be optional. To construct non-trivial input one can use the input_constructor argument of the get_model_complexity_info. input_constructor is a function that takes the input spatial resolution as a tuple and returns a dict with named input arguments of the model. Next this dict would be passed to the model as keyworded arguments.
verbose parameter allows to get information about modules that don't contribute to the final numbers.
ignore_modules option forces ptflops to ignore the listed modules. This can be useful for research purposes. For an instance, one can drop all convolutuions from the counting process specifying ignore_modules=[torch.nn.Conv2d].
'''
#python utils/get_flop.py
'''
Benchmark
torchvision
Model	Input Resolution	Params(M)	MACs(G)	Top-1 error	Top-5 error
alexnet	224x224	61.1	0.72	43.45	20.91
vgg11	224x224	132.86	7.63	30.98	11.37
vgg13	224x224	133.05	11.34	30.07	10.75
vgg16	224x224	138.36	15.5	28.41	9.62
vgg19	224x224	143.67	19.67	27.62	9.12
vgg11_bn	224x224	132.87	7.64	29.62	10.19
vgg13_bn	224x224	133.05	11.36	28.45	9.63
vgg16_bn	224x224	138.37	15.53	26.63	8.50
vgg19_bn	224x224	143.68	19.7	25.76	8.15
resnet18	224x224	11.69	1.82	30.24	10.92
resnet34	224x224	21.8	3.68	26.70	8.58
resnet50	224x224	25.56	4.12	23.85	7.13
resnet101	224x224	44.55	7.85	22.63	6.44
resnet152	224x224	60.19	11.58	21.69	5.94
squeezenet1_0	224x224	1.25	0.83	41.90	19.58
squeezenet1_1	224x224	1.24	0.36	41.81	19.38
densenet121	224x224	7.98	2.88	25.35	7.83
densenet169	224x224	14.15	3.42	24.00	7.00
densenet201	224x224	20.01	4.37	22.80	6.43
densenet161	224x224	28.68	7.82	22.35	6.20
inception_v3	224x224	27.16	2.85	22.55	6.44
'''