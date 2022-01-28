# CvT2DistilGPT2
#### Improving Chest X-Ray Report Generation by Leveraging Warm-Starting
- This repository houses the implementation of CvT2DistilGPT2 from [[1]](https://arxiv.org/abs/2201.09405).
- CvT2DistilGPT2 is an encoder-to-decoder model that was developed for chest X-ray report generation. 
- Checkpoints for CvT2DistilGPT2 on MIMIC-CXR and IU X-Ray are available.
- This implementation could be adapted for any image captioning task by modifying the datamodule.


|![](docs/figure.png)|
|----|
| <p align="center"> <a>CvT2DistilGPT2 for MIMIC-CXR. Q, K, and V are the queries, keys, and values, respectively, for multi-head attention. * indicates that the linear layers for Q, K, and V are replaced with the convolutional layers depicted below the multi-head attention module. `[BOS]` is the beginning-of-sentence special token. `N_l` is the number of layers for each stage, where `N_l=1`, `N_l=4`, and `N_l=16` for the first, second, and third stage, respectively. The head for DistilGPT2 is the same used for language modelling. Subwords produced by DistilGPT2 are separated by a vertical bar.</a> </p> |


## Installation
The required packages are located in `requirements.txt`. It is recommended that these are installed in a `virtualenv`:
```shell script
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt --no-cache-dir
```

## Datasets   

#### For MIMIC-CXR: 
1. Download MIMIC-CXR-JPG from: 
    ```
    https://physionet.org/content/mimic-cxr/2.0.0/
    ```
2. Place in `dataset/mimic_cxr_jpg` such that `dataset/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files`.

3. Download the [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf) labels for MIMIC-CXR from:
    ```
    https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing
    ```
4. Place `annotations.json` in `dataset/mimic_cxr_chen`

#### For IU X-Ray: 

1. Download the [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf) labels for IU X-Ray and the chest X-rays in `png` format from:
    ```
    https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view
    ```
4. Place files into `dataset/iu_x-ray_chen` such that `dataset/iu_x-ray_chen/annotations.json` and `dataset/iu_x-ray_chen/images`.

## Checkpoints   


## Instructions   
For the MIMIC-CXR (labels from [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf)):
```shell script
python3 main.py --task mimic_cxr_jpg_chen
``` 

For IU X-Ray (labels from [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf)):
```shell script
python3 main.py --task iu_x_ray_chen
``` 

## Reference
[1] [Aaron Nicolson, Jason Dowling, and Aaron Nicolson, **Improving Chest X-Ray Report Generation by Leveraging Warm-Starting**, Under review (January 2022)](https://arxiv.org/abs/2201.09405)



