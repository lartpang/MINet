# MINet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/lartpang/MINet?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/lartpang/MINet?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/lartpang/MINet?style=flat-square)
[![CVPR Page](https://img.shields.io/badge/CVPR%202020-MINet-blue?style=flat-square)](https://openaccess.thecvf.com/content_CVPR_2020/html/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.html)

## Folders & Files

* `backbone`: Store some code for backbone networks.
* `loss`: The code of the loss function.
* `module`: The code of important modules.
* `network`: The code of the network.
* `output`: It saves all results.
* `utils`: Some instrumental code.
    * `utils/config.py`: Configuration file for model training and testing.
    * `utils/imgs/*py`: Some files about creating the dataloader.
* `main.py`: I think you can understand.

## My Environment

* Python=3.7
* PyTorch=1.3.1 (I think `>=1.1` is OK. You can try it, please create an issue if there is a problem :smile:.)
* tqdm
* prefetch-generator
* tensorboardX

Recommended way to install these packages:

```
# create env
conda create -n pt13 python=3.7
conda activate pt13

# install pytorch
conda install pytorch=1.3.1 torchvision cudatoolkit=9.0 cudnn -c pytorch

# some tools
pip install tqdm
# (optional) https://github.com/Lyken17/pytorch-OpCounter
# pip install thop
pip install prefetch-generator

# install tensorboard
pip install tensorboardX
pip install tensorflow==1.13.1
```

## Note

* (2020-06-21 update) There are two important configuration items in the configuration dictionary: 
    * `exp_name`: It is divided into two parts by `@`. The first part is the class name of the corresponding model, and the second part is the additional string customized to distinguish between different configurations. 
        * For example, if we can set it to `MINet_VGG16@e_40_lr_0.025_opti_f3trick_sche_Poly`, we will use the model `MINet_VGG16` from the folder `network` (Note: you must import the model into the `./network/__init__.py` in advance), and the corresponding folder, where all files generated during the running of the model will be saved, will have the name `MINet_VGG16_e_40_lr_0.025_opti_f3trick_sche_Poly`.
    * `resume_mode`: It indicates whether you want to use the existing parameter files ('XXX.pth') stored in the `pth` folder in the model folder corresponding to `exp_name`. 
        * If you set it to `test`, and your `exp_name` is `MINet_VGG16@e_40_lr_0.025_opti_f3trick_sche_Poly`, the `.pth` file should be placed in `./output/MINet_VGG16_e_40_lr_0.025_opti_f3trick_sche_Poly/pth/state_final.pth` (**Please be careful to change the name of the `.pth` file to `state_final.pth` if you use our pretrained parameter file.**).
        * If your training process is interrupted, you can set it to `train`, and it will automatically load the existing `checkpoint_final.pth.tar` file from the pth folder to resume the training process.
        * In the case of **training the model from scratch** (that is, start_epoch=0), you just need to set it to `""`.

## Train

1. You can customize the value of the [`arg_config`](./utils/config.py#L20) dictionary in the configuration file.
    * The first time you use it, you need to adjust the [path](./utils/config.py#L9-L17) of every dataset.
    * Set the item `resume_mode` to `""`.
    * And other setting in `config.py`, like `exp_name`, `epoch_num`, `lr` and so on...
2. In the folder `code`, run the command `python main.py`.
3. Everything is OK. Just wait for the results.
4. The test will be performed automatically when the training is completed.
5. All results will be saved into the folder `output`, including predictions in folder `pre` (if you set `save_pre` to `True`), `.pth` files in folder `pth` and other log files.

## If you want to **test** the trained model again...

**Our pre-training parameters can also be used in this way.**

1. In the `output` folder, please ensure that there is **a folder corresponding to the model (See [Note](#Note))**, which contains the `pth` folder, and the `.pth` file of the model is located here and its name is `state_final.pth`.
2. Set the value of `exp_name` of `arg_config` to the model you want to test like these [lines](utils/config.py#L27-L30).
3. Set the value of `te_data_list` to your dataset path.
4. Set the value of `resume_mode` to `test`.
5. In the folder `code`, run `python main.py`.
6. You can find predictions from the model in the folder `pre` of the `output`.

## Evaluation

We evaluate results of all models by [lartpang/sal_eval_toolbox](https://github.com/lartpang/SODEvalToolkit/tree/master/tools).

> [lartpang/sal_eval_toolbox](https://github.com/lartpang/SODEvalToolkit/tree/master/tools) is based on [ArcherFMY/sal_eval_toolbox](https://github.com/ArcherFMY/sal_eval_toolbox/tree/master/tools).
>
> But, we add the code about E-measure and weighted F-measure and update the related code in our forked repository [lartpang/sal_eval_toolbox](https://github.com/lartpang/SODEvalToolkit/tree/master/tools). Welcome to use it in your code :star:!

## More

If there are other issues, you can create a new issue.

## More Experiments

F3Net is the most recent work on SOD, and the performance is very good. I think its training strategy is of great reference value. Here, I have changed our training method by learning from its code.

To explore the upper limit of the performance of the model, I tried some ways to improve the performance of the model on a NVIDIA GTX 1080Ti (~11G). 
* To achieve a larger batch size:
    * we reduce the number of intermediate channels in AIMs;
    * we apply the `checkpoint` feature of PyTroch;
        * <https://blog.csdn.net/one_six_mix/article/details/93937091>
        * <https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint>
    * we set the batch size to 32.
* Use more effective training strategies:
    * We apply multi-scale training strategy (borrowed from the code of F3Net).
    * Network parameters are trained in groups with different learning rates (borrowed from the code of F3Net).
    * Multiple learning rate decay strategies with/without the warmup technology.
    
Results:

 D    |         |         | DO |         |         | HI |         |         | E   |         |         | PS |         |         | SOC     |         |         | MT | LRDecay                 | Optimizer | InitLR | Scale | EP 
---------|---------|---------|------------|---------|---------|---------|---------|---------|---------|---------|---------|-----------|---------|---------|---------|---------|---------|---------------|-------------------------|-----------------|----------|-----------|-------
 MAXF    | MEANF   | MAE     | MAXF       | MEANF   | MAE     | MAXF    | MEANF   | MAE     | MAXF    | MEANF   | MAE     | MAXF      | MEANF   | MAE     | MAXF    | MEANF   | MAE     |               |                         |                 |          |           |       
 0\.853  | 0\.787  | 0\.048  | 0\.794     | 0\.734  | 0\.060  | 0\.922  | 0\.891  | 0\.036  | 0\.931  | 0\.908  | 0\.043  | 0\.856    | 0\.810  | 0\.084  | 0\.377  | 0\.342  | 0\.086  | FALSE         | Poly                    | Sgd\_trick      | 0\.05    | 320       | 40    
 0\.866  | 0\.793  | 0\.043  | 0\.789     | 0\.722  | 0\.059  | 0\.925  | 0\.888  | 0\.034  | 0\.935  | 0\.905  | 0\.041  | 0\.874    | 0\.822  | 0\.070  | 0\.382  | 0\.347  | 0\.110  | FALSE         | Poly                    | Sgd\_trick      | 0\.001   | 320       | 40    
 0\.881  | 0\.822  | 0\.037  | 0\.803     | 0\.746  | 0\.053  | 0\.934  | 0\.904  | 0\.029  | 0\.942  | 0\.919  | 0\.036  | 0\.880    | 0\.837  | 0\.066  | 0\.390  | 0\.356  | 0\.081  | FALSE         | Poly                    | Sgd\_trick      | 0\.005   | 320       | 40    
 0\.878  | 0\.815  | 0\.039  | 0\.803     | 0\.745  | 0\.054  | 0\.934  | 0\.904  | 0\.029  | 0\.944  | 0\.919  | 0\.035  | 0\.878    | 0\.833  | 0\.067  | 0\.385  | 0\.352  | 0\.079  | FALSE         | Cos\_warmup          | Sgd\_trick      | 0\.005   | 320       | 40    
 0\.878  | 0\.815  | 0\.038  | 0\.797     | 0\.741  | 0\.054  | 0\.931  | 0\.901  | 0\.031  | 0\.941  | 0\.917  | 0\.038  | 0\.875    | 0\.831  | 0\.067  | 0\.382  | 0\.355  | 0\.085  | FALSE         | Cos\_warmup          | Sgd\_trick      | 0\.003   | 320       | 40    
 0\.892  | 0\.836  | 0\.036  | 0\.820     | 0\.763  | 0\.053  | 0\.943  | 0\.918  | 0\.026  | 0\.950  | 0\.929  | 0\.034  | 0\.884    | 0\.847  | 0\.064  | 0\.388  | 0\.355  | 0\.087  | TRUE          | f3\_sche                | f3\_trick       | 0\.05    | 352       | 40    
 0\.891  | 0\.834  | 0\.037  | 0\.820     | 0\.762  | 0\.055  | 0\.942  | 0\.915  | 0\.026  | 0\.948  | 0\.928  | 0\.034  | 0\.888    | 0\.844  | 0\.064  | 0\.394  | 0\.359  | 0\.120  | TRUE          | Cos\_warmup          | f3\_trick       | 0\.05    | 352       | 40    
 0\.895  | 0\.840  | 0\.035  | 0\.816     | 0\.762  | 0\.055  | 0\.942  | 0\.915  | 0\.027  | 0\.947  | 0\.927  | 0\.034  | 0\.884    | 0\.843  | 0\.066  | 0\.395  | 0\.359  | 0\.112  | TRUE          | Cos\_w/o\_warmup | f3\_trick       | 0\.05    | 352       | 40    
 0\.893  | 0\.838  | 0\.036  | 0\.814     | 0\.759  | 0\.056  | 0\.943  | 0\.917  | 0\.026  | 0\.949  | 0\.930  | 0\.033  | 0\.886    | 0\.849  | 0\.065  | 0\.395  | 0\.359  | 0\.134  | TRUE          | Poly                    | f3\_trick       | 0\.05    | 352       | 40    

* D: DUTS
* DO: DUT-OMRON
* HI: HKU-IS
* E: ECSSD
* PS: PASCAL-S
* MT: Multi-scale Training
* EP: Epoch Number

NOTE: The results here are for reference only. Note that the results here are all tested on the complete test datasets. In fact, some of the results here can be higher if testing in the way of the existing papers. Because the test set in my paper follows the settings of the existing papers, some datasets, such as HKU-IS, are not tested with the complete dataset.

注：
此处结果仅供参考。请注意，这里的结果都是在完整的测试数据集上测试的。事实上，如果按照现有论文的方式进行测试，这里的一些结果可能会更高。由于本文中的测试集遵循现有论文的设置，一些数据集，如HKU-IS，没有使用完整的数据集进行测试。
