# Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation

<div align="center"> Yujie Zhang<sup>1*</sup>, Bingyang Cui<sup>1*</sup>, Qi Yang<sup>2</sup>, Zhu Li<sup>2</sup>, Yiling Xu<sup>1</sup>

<div align="center"> <small>1 Shanghai Jiao Tong University, 2 University of Missouri-Kansas City</small>
<div align="center"> <small><sup>*</sup> Indicates Equal Contribution</small>
<p align="center">
  <a href="http://arxiv.org/abs/" target='_**blank**'>
    <img src="https://img.shields.io/badge/arXiv paper-2312.02980📖-blue?">
  </a> 
  <a href="https://mate-3d.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
</p>
</div>
</div>
</div>

## 🎦 Introduction

Text-to-3D generation has achieved remarkable progress in recent years, yet evaluating these methods remains challenging for two reasons: i) Existing benchmarks lack fine-grained evaluation on different prompt categories and evaluation dimensions. ii) Previous evaluation metrics only focus on a single aspect (e.g., text-3D alignment) and fail to perform multi-dimensional quality assessment. To address these problems, we first propose a comprehensive benchmark named **MATE-3D**. The benchmark contains eight well-designed prompt categories that cover single and multiple object generation, resulting in 1,280 generated textured meshes. We have conducted a large-scale subjective experiment from four different evaluation dimensions and collected 107,520 annotations, followed by detailed analyses of the results. Based on MATE-3D, we propose a novel quality evaluator named **HyperScore**. Utilizing hypernetwork to generate specified mapping functions for each evaluation dimension, our metric can effectively perform multi-dimensional quality assessment. HyperScore presents superior performance over existing metrics on MATE-3D, making it a promising metric for assessing and improving text-to-3D generation.

<div align="center">
<img src="https://github.com/zhangyujie-1998/MATE-3D/blob/main/asset/framework.jpg" width = 80% height = 80%/>
<br>
Overview of the HyperScore Evaluator
</div>

## 📦 Dataset Preparation

**NOTE:** Since the dataset used in our training is based on MATE-3D, please first download MATE-3D dataset from [onedrive](https://1drv.ms/u/c/669676c02328fc1b/EdJ0J23NWOZOprClaz4pKjQBEp-V-fVFQ7FAT2vZoZsbJw?e=qXgIwt) or  [huggingface](https://huggingface.co/datasets/ccccby/MATE-3D) and unzip  it into ```data``` folder. The file structure of used data should be like:

```
-data
  --MATE-3D
    ---3dtopia
      ----A_badge_shaped_like_a_shield 
      ----A_bat_is_hanging_upside_down_from_a_branch_with_its_wings_folded
      ...
    ---consistent3d
    ---dreamfusion
    ---latentnerf
    ---sjc
    ---textmesh
    ---magic3d
    ---one2345++
    ---prompt_MATE_3D.json
    ---prompt_MATE_3D.xlsx
```

## 🔧 Installation

Please use the following commands to install dependencies:

```
conda create --name HyperScore python=3.10
conda activate HyperScore 
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

Additionally, we render texture meshed into images by Pytorch3D, please follow the steps to install Pytorch3D.

```
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

## 🚆 Training

We implement our metric by PyTorch and conduct training and testing on the NVIDIA 3090GPUs.

**NOTE:** Since the dataset used in our training is based on MATE-3D, please first prepare MATE-3D dataset in  the ```data``` folder. Then, you need to render the texture meshes into multi-view images by

 ```
python get_projection.py
 ```

The multi-view images will be saved in the ```data/projection``` folder.

- Now you can start to train the model as follows, and the results will be restored in the ```results``` folder.

 ```
 bash train.sh
 ```

## 🏁 Demo

You can use demo.py to predict the quality of one  textured mesh. You need to first download the checkpoint from [onedrive](https://1drv.ms/u/c/669676c02328fc1b/EbUs_rWDXtREoXW_brOk_bkBzdFM6hyxFUoevRhRj1Zxmw?e=l4gIgs) and put it into the ```checkpoint``` folder.  Then, you can run 

```
python demo.py

# example: prompt is "A canned Coke"
# obj_path = "demo/A_canned_Coke/model.obj"
```

If you want to infer other textured mesh, please edit  the ' obj_path'  and 'prompt' in the demo.py. 

## 📖 Citation

If you find this work is helpful, please consider citing:

```bash
@article{,
  title={Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation},
  author={Yujie Zhang, Bingyang Cui, Qi Yang, Zhu Li, and Yiling Xu},
  journal={arXiv preprint arXiv:2},
  year={2024}
}
```

