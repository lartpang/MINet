# MINet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/lartpang/MINet?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/lartpang/MINet?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/lartpang/MINet?style=flat-square)
[![CVPR Page](https://img.shields.io/badge/CVPR%202020-MINet-blue?style=flat-square)](https://openaccess.thecvf.com/content_CVPR_2020/html/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.html)

## Folders & Files

<details>
<summary>Directory Structure</summary>

```shell script
$ tree -L 3
.
├── backbone
│   ├── __init__.py
│   ├── origin
│   │   ├── from_origin.py
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   └── wsgn
│       ├── customized_func.py
│       ├── from_wsgn.py
│       ├── __init__.py
│       ├── resnet.py
│       └── resnext.py
├── config.py
├── LICENSE
├── loss
│   ├── CEL.py
│   └── __init__.py
├── main.py
├── module
│   ├── BaseBlocks.py
│   ├── __init__.py
│   ├── MyLightModule.py
│   ├── MyModule.py
│   └── WSGNLightModule.py
├── network
│   ├── __init__.py
│   ├── LightMINet.py
│   ├── MINet.py
│   ├── PureWSGNLightMINet.py
│   └── WSGNLightMINet.py
├── output (These are the files generated when I ran the code.)
│   ├── CPLightMINet_Res50_S352_BS32_E20_WE1_AMPy_LR0.05_LTf3sche_OTf3trick_ALy_BIy_MSy
│   │   ├── cfg_2020-07-23.txt
│   │   ├── pre
│   │   ├── pth
│   │   ├── tb
│   │   ├── te_2020-07-23.txt
│   │   ├── tr_2020-07-23.txt
│   │   └── trainer_2020-07-23.txt
│   └── result.xlsx
├── pyproject.toml
├── readme.md
└── utils
    ├── cal_fps.py
    ├── dataloader.py
    ├── __init__.py
    ├── joint_transforms.py
    ├── metric.py
    ├── misc.py
    ├── pipeline_ops.py
    ├── recorder.py
    ├── solver.py
    └── tensor_ops.py

```
</details>

* `backbone`: Store some code for backbone networks.
* `loss`: The code of the loss function.
* `module`: The code of important modules.
* `network`: The code of the network.
* `output`: It saves all results.
* `utils`: Some instrumental code.
    * `dataloader.py`: About creating the dataloader.
    * ...
* `main.py`: I think you can understand.
* `config.py`: Configuration file for model training and testing.

## My Environment

**Latest Env Info**

The yaml file exported by my latest work environment is `code/ minet.yaml`. You can refer to the version information of each package in it.

For Apex:

```shell script
$ # For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions via
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Note

<details>
<summary>Configraturn Demo</summary>

```python
# CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn
arg_config = {
    "model": "CPLightMINet_Res50",  # 实际使用的模型，需要在`network/__init__.py`中导入
    "info": "",  # 关于本次实验的额外信息说明，这个会附加到本次试验的exp_name的结尾，如果为空，则不会附加内容。
    "use_amp": False,  # 是否使用amp加速训练
    "resume_mode": "",  # the mode for resume parameters: ['train', 'test', '']
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": False,  # 是否保留最终的预测结果
    "epoch_num": 50,  # 训练周期, 0: directly test model
    "lr": 0.001,  # 微调时缩小100倍
    "xlsx_name": "result.xlsx",  # the name of the record file
    # 数据集设置
    "rgb_data": {
        "tr_data_path": dutstr_path,
        "te_data_list": OrderedDict(
            {
                "pascal-s": pascals_path,
                "ecssd": ecssd_path,
                # "hku-is": hkuis_path,
                # "duts": dutste_path,
                # "dut-omron": dutomron_path,
                # "soc": soc_path,
            },
        ),
    },
    # 训练过程中的监控信息
    "tb_update": 10,  # >0 则使用tensorboard
    "print_freq": 10,  # >0, 保存迭代过程中的信息
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名
    "prefix": (".jpg", ".png"),
    # if you dont use the multi-scale training, you can set 'size_list': None
    # "size_list": [224, 256, 288, 320, 352],
    "size_list": None,  # 不使用多尺度训练
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    # 优化器与学习率衰减
    "optim": "f3_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,  # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 训练时是否对真值二值化（阈值为0.5）
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 320,
}
```

</details>

* (2020-07-23 update) There are some important configuration items in the configuration system: 
    * `model` in `config.py`: It is the name of the model that will be used. (Note: you must import the model into the `./network/__init__.py` in advance.)
        * For example, if we can set `model` to `MINet_VGG16`, we will use the model `MINet_VGG16` from the folder `network`.
    * `info` in `config.py`: An additional string used to illustrate this experiment.
    * `exp_name` in `main.py`: The complete string used to describe this experiment, which is obtained by the function function `construct_exp_name` in `utils/misc.py` by parsing `config.py`. For more information, please see `utils/ misc.py`.
        * Based on the demo configuration, the `exp_name` generated by our code is `CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn`.  
        * **The corresponding folder in `output`**, where all files generated during the running of the model will be saved, will have the name `CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn`.
    * `resume_mode`: It indicates whether you want to use the existing parameter files ('XXX.pth') stored in the `pth` folder in the model folder corresponding to `exp_name`. 
        * If you set it to `test`, and your `exp_name` is `CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn`, the `.pth` file should be placed in `./output/CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn` (**Please be careful to change the name of the `.pth` file to `state_final.pth` if you use our pretrained parameter file.**).
        * If your training process is interrupted, you can set it to `train`, and it will automatically load the existing `checkpoint_final.pth.tar` file from the pth folder to resume the training process.
        * In the case of **training the model from scratch** (that is, start_epoch=0), you just need to set it to `""`.

## Train

1. You can customize the value of the [`arg_config`](config.py#L20) dictionary in the configuration file.
    * The first time you use it, you need to adjust the [path](config.py#L9-L17) of every dataset.
    * Set `model` to the model name that has been imported in `network/__init__.py`.
    * Modify `info` according to your needs, which will be appended to the final `exp_name`. (The function `construct_exp_name` in `utils/misc.py` will generate the final `exp_name`.)
    * Set the item `resume_mode` to `""`.
    * And other setting in `config.py`, like `epoch_num`, `lr` and so on...
2. In the folder `code`, run the command `python main.py`.
3. Everything is OK. Just wait for the results.
4. The test will be performed automatically when the training is completed.
5. All results will be saved into the folder `output`, including predictions in folder `pre` (if you set `save_pre` to `True`), `.pth` files in folder `pth` and other log files.

## If you want to **test** the trained model again...

**Our pre-training parameters can also be used in this way.**

1. In the `output` folder, please ensure that there is **a folder corresponding to the model (See [Note](#Note))**, which contains the `pth` folder, and the `.pth` file of the model is located here and its name is `state_final.pth`.
2. Set the value of `model` of `arg_config` to the model you want to test.
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

<details>

<summary>More Experiments</summary>

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

</details>
