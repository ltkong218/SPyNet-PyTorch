# SPyNet PyTorch

PyTorch implementation of Spatial Pyramid Network. Original paper: [Optical Flow Estimation using a Spatial Pyramid Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ranjan_Optical_Flow_Estimation_CVPR_2017_paper.pdf).

****	
|Author|Tone|
|---|---
|E-mail|ltkong218@gmail.com
****


## Introduction and something to know
My trained models are in './models/myClean'.

When you want to use the orginal paper models converted from lua in './models/* /*.t7', you should add '--use_pytorch_model False', and set the 'model_path' like '--mode_path './models/chairsClean''.

With PyTorch version == 0.4.0, when you use --num_workers > 1, it will have some warnings, but this doesn't affect training and testing, I will research it later.

When you use data parallel to train a model and want to use it in other pc without data parallel, you should remove 'module' in the parameters name of the parallel model by using './convert_model.ipynb'.


## Requirement
pytorch-0.4.0, torchvision, visdom and so on.


## Demo
For showing optical flow result, you can simply use

```Shell
python spynet.py
```
**img1**  
![img1](https://github.com/ltkong218/SPyNet-PyTorch/raw/master/data/img1.jpg "img1")

**img2**  
![img2](https://github.com/ltkong218/SPyNet-PyTorch/raw/master/data/img2.jpg "img2")

**optical flow**  
![flow](https://github.com/ltkong218/SPyNet-PyTorch/raw/master/eval_result/flow.png "flow")


## Train
For training, you can refer to the following example

```Shell
python main.py --level 1 --num_epoches 1 --use_pretrained False
```

```Shell
python main.py --use_visdom True --checkpoint_path './checkpoint' --result_path './result' --num_epoches 90 --use_pretrained False --cycle_train False --optim_method 'Adam' --lr_scheduler 'MultiStepLR' --milestones '30' --is_parallel True --parallel_gpu_ids = '1,2,3' --level 5 --aug_make_data_gpu_id 0 --lr 5e-5 --batch_size 64 --is_augment True --angle 0 --scale 8 --noise 0 --brightness 0 --contrast 0 --saturation 0 --lighting 0
```

When set '--use_pretrained True', you should add the current model in root_dir and rename it as 'model_pretrained.pth' or 'model_pretrained.t7'.


## Test benchmark
For testing my trained models benchmark, you should copy './models/myClean/model5.pth' to the root_dir, and rename it as 'model_pretrained.pth'.

```Shell
run python main.py --mode 'test' --level 5 --num_workers 0 --batch_size 8
```

For testing other lua models benchmark, you should copy './models/chairsClean/model5.t7' to the root_dir and rename it as 'model_pretrained.t7'.
Run

```Shell
python main.py --mode 'test' --level 5 --num_workers 0 --batch_size 8 --model_path './models/chairsClean' --use_pytorch_model False
```


## Some results

****
| **chairsClean** | **EPE:2.657068** |
| ---------- | -----------|
| **chairsFinal** | **EPE:2.795757**
| **myClean** | **EPE:2.659885** |
****


With more training time, we can get lower EPE.

## Training details 
myClean level1, level2, level3 are trained on 2 gtx1080ti, one is used to make and augment data with multiprocessing, the other is used to train the current level model.
You can use the default augment parameters to train first, but to reach the benchmark, you should set the parameters like

```Shell
--angle 0 --scale 8 --noise 0 --brightness 0 --contrast 0 --saturation 0 --lighting 0
```

myClean level4, level5 are trained on 4 gtx1080ti, one is used to make and augment data with multiprocessing, the other three are used to train the current level model in data parallel mode.
For the higher levels are more hard to convergence, we don't use the default augment parameters. To reach the benchmark, for model4 and model5 respectively you can use

```Shell
--angle 0 --scale 5 --noise 0 --brightness 0 --contrast 0 --saturation 0 --lighting 0
```

```Shell
--angle 0 --scale 3 --noise 0 --brightness 0 --contrast 0 --saturation 0 --lighting 0
```

'--lr' first set to 1e-4, then change to the range from 1e-5 to 2e-6 according to the situation.

level1 for about 3 days. level2, level3 for about 2 days. level4, level5 for about 2.5 days.
