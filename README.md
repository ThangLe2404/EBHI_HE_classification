# EBHI_HE_classification
This repository contains code for our study:

"Deep Learning-Driven Subtype Classification of Colorectal Cancer Using Histopathology Images for Personalized Medicine"

doi: https://doi.org/10.1101/2024.12.12.628270


# Setup 

Step 1. Download and install miniconda following the [official instruction](https://www.anaconda.com/docs/getting-started/miniconda/main)

Step 2. Create new conda environment

```
conda create --name ebhi-seg python=3.11 -y
conda activate ebhi-seg
```

Step 3. Select and install a pytorch version following the [official website](https://pytorch.org/get-started/locally/). For example, in our local machine, we used pytorch 2.5.1 with cuda12.1:

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Step 4. Install pytorch lightning

```
conda install lightning -c conda-forge
```

Step 5. Install timm for using pretrained models

```
pip install timm
```


Step 6. Install imgaug for data augmentation

```
conda install imgaug
```

Step 7. Install sklearn

```
pip install scikit-learn
```
Note: if you get numpy error, replace numpy 2 by numpy 1.26

## For evaluation

Install seaborn for figure and diagram visualization

```
pip install seaborn
```

Install thop and torchinfo for models summarization

```
pip install thop -y
pip install torchinfo 
```

# Training 

Can add more architecture by adding more element to models_type dictionary in [Training.py](https://github.com/ThangLe2404/EBHI_HE_classification/blob/main/Trainning.py) file. Please check [timm](https://huggingface.co/timm)'s models list.

```python
models_type = {
    'resnet18': 'resnet18.a1_in1k',
    'resnet34': 'resnet34.a1_in1k',
    'resnet50': 'resnet50.a1_in1k',
    'swinv2_t_w8_256': 'swinv2_tiny_window8_256.ms_in1k',
    'swinv2_t_w16_256': 'swinv2_tiny_window16_256.ms_in1k',
    'swinv2_s_w8_256': 'swinv2_small_window8_256.ms_in1k',
    'swinv2_s_w16_256': 'swinv2_small_window16_256.ms_in1k',
    }
```

Then call `main()` function with key you use the dictionary and adjust the params such as val_path, train_path, test_path, optimizer, learning rate, etc. E.g

```
model = main('swinv2_s_w16_256', num_epochs=100, batch_size=32)
```

# Evaluation and Gradcam

Similar to training, you can also add new model type by changing `models_type` dictionary.

You will need to modify training checkpoint paths in weight_paths to evaluate.

```
weight_paths = {
    'resnet18': ['./resnet18-epoch=8-val_loss=0.27-val_acc=0.91.ckpt'],
    'resnet34': [#'./resnet34-epoch=8-val_loss=0.35-val_acc=0.91.ckpt', 
                 './resnet34-epoch=11-val_loss=0.36-val_acc=0.88.ckpt'],# './resnet34-epoch=11-val_loss=0.36-val_acc=0.88.ckpt'],
    'resnet50': [#'./resnet50-epoch=29-val_loss=0.37-val_acc=0.90.ckpt', 
                 './resnet50-epoch=29-val_loss=0.37-val_acc=0.90.ckpt'],
'swinv2_t_w8_256': ['./swinv2_t_w8_256-epoch=66-val_loss=0.41-val_acc=0.88.ckpt'] ,#,'./swinv2_t_w8_256-epoch=30-val_loss=0.42-val_acc=0.87.ckpt'],
    #'swinv2_t_w16_256': ['./swinv2_t_w16_256-epoch=46-val_loss=0.45-val_acc=0.89.ckpt','./swinv2_t_w16_256-epoch=34-val_loss=0.42-val_acc=0.88.ckpt'],
    'swinv2_s_w8_256': [#'./swinv2_s_w8_256-epoch=86-val_loss=0.47-val_acc=0.89.ckpt', './swinv2_s_w8_256-epoch=35-val_loss=0.38-val_acc=0.87.ckpt', 
                        './swinv2_s_w8_256-epoch=31-val_loss=0.47-val_acc=0.86.ckpt'
                        ]
    #'swinv2_s_w16_256': ['./swinv2_s_w8_256-epoch=86-val_loss=0.47-val_acc=0.89.ckpt'],
    }

```

Install [pytorch gradcam](https://github.com/jacobgil/pytorch-grad-cam) library

```
pip install grad-cam
```

Then run the evaluation or gradcam

```
python Evaluation.py
or
python Gradcam.py
```

Note: change model_name and weight_path variable in Gradcam.py file 

# Data Preparation
In our study, we use [EBHI-Seg](https://www.kaggle.com/datasets/lavensrivastava/ebhi-seg) dataset, contains 5,170 H&E-stained images annotated across 6 CRC subtypes.

# Data augmentation 

Please modify these variables to your local paths

```
raw_dataset_path = '/content/drive/MyDrive/EBHI-SEG'
dataset_path = '/content/drive/MyDrive/EBHI-SEG/dataset'
augment_dataset_path = './'
```
