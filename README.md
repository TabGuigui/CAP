# CAP

implementation of paper "Constrained Adaptive Projection with Pretrained Features for Anomaly Detection“ （CAP）

paper address: https://arxiv.org/abs/2112.02597 (not final version)



### Framework

<div align="center">    
    <img src="pics/framework.png">  
</div>

## 

### Experiment

##

#### quickly run

> For cifar10
```
sh cifar10.sh
```
> For cifar100, exchange --dataset cifar100

> For mvTec, please download mvTec dataset and exchange hyperparameters corresponding to appendix file.


#### run a specific class

```
python main.py --dataset <dataset> --normal_class <normal-class> --regular <constrained lambda>
```



