# ST-AT
Code for our ICLR 2023 paper [Squeeze Training for Adversarial Robustness](https://openreview.net/pdf?id=Z_tmYu060Kr).

## Environments
* Python 3.6.8
* numpy 1.19.5
* pandas 1.1.5
* Pillow 9.0.0
* scipy 1.5.4
* statsmodels 0.12.2
* torch 1.8.0+cu111
* torchvision 0.9.0+cu111

## Datasets
Datasets should be prepared into the ```st-at/data```.

## Train
Train a ResNet-18 on CIFAR-10/CIFAR-100/SVHN using our ST:
```
dataset=${dataset} beta=${beta} epochs=${epochs} lossmode=${lossmode} bash st.sh
```
where ```${dataset}``` in ```["cifar10", "cifar100", "svhn"]```, ```${epochs}``` in ```["120", "80"]```, and ```${lossmode}``` in ```["js","l2","symmkl"]```.

Train a WRN-28-10 on CIFAR-10 using ST-RST:
```
bash st_rst.sh
```
## Test
***FGSM, PGD and CW for ST trained ResNet-18***:
```
python3 test.py --model-path ${modelpath} --log ${logpath} --dataset ${dataset}
```
***PGD<sub>RST</sub> for ST-RST trained WRN-28-10***:
```  
python3 test.py --model_path ${modelpath} --attack pgd --output_suffix pgd_rst
```
***AutoAttack***
```  
python3 test_aa.py --model-path ${modelpath} --arch ${architecture} --dataset ${dataset} --log ${logpath}
```
where ```${architecture}``` in ```["resnet18", "wrn-28-10"]```.

## Citation
Please cite our work in your publications if it helps your research:

```
@article{li2023squeeze,
  title={Squeeze Training for Adversarial Robustness},
  author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
  booktitle={ICLR},
  year={2023}
}
```
