# MINet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/lartpang/MINet?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/lartpang/MINet?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/lartpang/MINet?style=flat-square)
[![CVPR Page](https://img.shields.io/badge/CVPR%202020-MINet-blue?style=flat-square)](https://openaccess.thecvf.com/content_CVPR_2020/html/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.html)
[![Arxiv Page](https://img.shields.io/badge/Arxiv-2007.09062-red?style=flat-square)](https://arxiv.org/abs/2007.09062)

## NOTE

**More Details can be found in the branch `master`**

- This branch is a **simplified version**. A complete version and more details can be found in the branch `master`.
- If there are other issues, you can create a new issue.
- In this branch, I use the code of calculating the metrics from <https://github.com/lartpang/PySODMetrics>, which is
  faster and Consistent with Fan's MATLAB code <https://github.com/DengPingFan/CODToolbox>. It's noted that some
  details are different from the `master` branch's code.

## Folders & Files

<details>
<summary>Directory Structure</summary>

```shell script
$ tree .
.
├── assets
│   ├── AIM.png
│   ├── CurveFigure.png
│   ├── Network.png
│   ├── SIM.png
│   ├── TableofResults.png
│   └── VisualFigure.png
├── config.py
├── LICENSE
├── main.py
├── minet.yaml
├── pyproject.toml
├── README.md
├── USAGE.md
└── utils
    ├── dataloader.py
    ├── dataset_configs.py
    ├── __init__.py
    ├── metrics.py
    ├── models.py
    ├── pipeline_ops.py
    └── tool_funcs.py
```

</details>

* `main.py` I think you can understand.
* `models.py` The complete MINet (VGG16/19).
* `utils` Some instrumental code.
* `config.py` Configuration file for model training and testing.
* `pipeline_ops.py` and `tool_funcs.py` Some useful tools for the train process.
* `metrics.py` The code about different metrics.

## My Environment

For Apex:

```shell script
$ # For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions via
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Note

* (2020-07-23 update) There are some important configuration items in the configuration system:
    * `model` in `config.py`: It is the name of the model that will be imported from the `model.py`.
        * For example, if we can set `model` to `MINet_VGG16`, we will use the model `MINet_VGG16` from the
          file `model.py`.
    * `info` in `config.py`: An additional string used to illustrate this experiment.
    * `exp_name` in `main.py`: The complete string used to describe this experiment, which is obtained by the function
      `construct_exp_name` in `utils.py` by parsing `config.py`. For more information, please see `utils.py` and the
      the `readme.md` of the `master` branch.
    * `resume_mode`: It indicates whether you want to use the existing parameter files ('XXX.pth') stored in the `pth`
      folder in the model folder corresponding to `exp_name`.
        * If you set it to `test`, and your `exp_name`
          is `CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn`, the `.pth` file should be
          placed in `./output/CPLightMINet_Res50_S320_BS4_E50_WE1_AMPn_LR0.001_LTpoly_OTf3trick_ALy_BIy_MSn` (**Please
          be careful to change the name of the `.pth` file to `state_final.pth` if you use our pretrained parameter
          file.**).
        * If your training process is interrupted, you can set it to `train`, and it will automatically load the
          existing ~~`checkpoint_final.pth.tar`~~ `checkpoint_final.pth` file from the pth folder to resume the
          training process.
        * In the case of **training the model from scratch** (that is, start_epoch=0), you just need to set it to `""`.

## Train

1. You can customize the value of the [`arg_config`](./config.py) dictionary in the configuration file.
    * The first time you use it, you need to adjust the `path` of every dataset.
    * Set `model` to the model name that has been imported in `model.py`.
    * Modify `info` according to your needs, which will be appended to the final `exp_name`. (The
      function `construct_exp_name` in `utils.py` will generate the final `exp_name`.)
    * Set the item `resume_mode` to `""`.
    * And other setting in `config.py`, like `epoch_num`, `lr` and so on...
2. Run the command `python main.py`.
3. Everything is OK. Just wait for the results.
4. The test will be performed automatically when the training is completed.
5. All results will be saved into the folder `output`, including predictions in folder `pre` (if you set `save_pre`
   to `True`), `.pth` files in folder `pth` and other log files.

## If you want to **test** the trained model again...

**Our pre-training parameters can also be used in this way.**

1. In the `output` folder, please ensure that there is **a folder corresponding to the model (See [Note](#Note))**,
   which contains the `pth` folder, and the `.pth` file of the model is located here and its name is `state_final.pth`.
2. Set the value of `model` of `arg_config` to the model you want to test.
3. Set the value of `te` to your dataset path.
4. Set the value of `resume_mode` to `test`.
5. Run `python main.py`.
6. You can find predictions from the model in the folder `pre` of the `output`.
