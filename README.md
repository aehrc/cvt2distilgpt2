# CvT2DistilGPT2
#### Improving Chest X-Ray Report Generation by Leveraging Warm-Starting
- This repository houses the code for CvT2DistilGPT2 from [[1]](https://arxiv.org/abs/2201.09405) and is implemented in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/).
- CvT2DistilGPT2 is an encoder-to-decoder model that was developed for chest X-ray report generation. 
- Its encoder is the Convolutional vision Transformer (CvT) warm-started with an ImageNet-21K checkpoint.
- Its decoder is DistilGPT2 (which describes the architecture of the decoder as well as the checkpoint).
- Checkpoints for CvT2DistilGPT2 on MIMIC-CXR and IU X-Ray are available.
- This implementation could be adapted for any image captioning task by modifying the datamodule.


|![](docs/figure.png)|
|----|
| <p align="center"> <a>CvT2DistilGPT2 for MIMIC-CXR. Q, K, and V are the queries, keys, and values, respectively, for multi-head attention. * indicates that the linear layers for Q, K, and V are replaced with the convolutional layers depicted below the multi-head attention module. `[BOS]` is the beginning-of-sentence special token. `N_l` is the number of layers for each stage, where `N_l=1`, `N_l=4`, and `N_l=16` for the first, second, and third stage, respectively. The head for DistilGPT2 is the same used for language modelling. Subwords produced by DistilGPT2 are separated by a vertical bar.</a> </p> |

## Installation
After cloning the repository, install the required packages in a virtual environment.
The required packages are located in `requirements.txt`:
```shell script
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt --no-cache-dir
```
Next, download the **datasets** and **checkpoints**, as describe below.

## Datasets   

### For MIMIC-CXR: 
1. Download MIMIC-CXR-JPG from: 
    ```
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    ```
2. Place the files in `dataset/mimic_cxr_jpg` so that the following path exists `dataset/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files`.

3. Download the [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf) labels for MIMIC-CXR from:
    ```
    https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing
    ```
4. Place `annotations.json` in `dataset/mimic_cxr_chen` such that its path is `dataset/mimic_cxr_chen/annotations.json`

### For IU X-Ray: 

1. Download the [Chen *et al.*](https://aclanthology.org/2020.emnlp-main.112.pdf) labels and the chest X-rays in `png` format for IU X-Ray from:
    ```
    https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view
    ```
2. Place the files into `dataset/iu_x-ray_chen` such that their paths are `dataset/iu_x-ray_chen/annotations.json` and `dataset/iu_x-ray_chen/images`.

##### Note: the `dataset` directory can be changed for each task with the variable `dataset_dir` in `task/mimic_cxr_jpg_chen/paths.yaml` and `task/mimic_cxr_jpg_chen/paths.yaml`

## Checkpoints
### CvT2DistilGPT2    
 CvT2DistilGPT2 checkpoints for MIMIC-CXR and IU X-Ray can be found at: [https://doi.org/10.25919/64wx-0950](https://doi.org/10.25919/64wx-0950) (click on the *files* tab to download individual checkpoints). 
 
 Place the checkpoints in the experiment directory for each model of each task, e.g., place the checkpoint:
  
 ![](docs/example.png)

  at the path: `experiment/mimic_cxr_jpg_chen/cvt_21_to_gpt2_scst/epoch=0-val_chen_cider=0.410965.ckpt`.
  
##### Note: the `experiment` directory can be changed for each task with the variable `exp_dir` in `task/mimic_cxr_jpg_chen/paths.yaml` and `task/iu_x_ray_chen/paths.yaml`

### CvT-21 Checkpoint

Download `CvT-21-384x384-IN-22k.pth` from this [Microsoft model zoo](https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712) and place it in `checkpoints` such that its path is `checkpoints/CvT-21-384x384-IN-22k.pth`

### DistilGPT2 Checkpoint

Download `config.json`, `tokenizer.json`, `pytorch_model.bin`, and `vocab.json` from [https://huggingface.co/distilgpt2/tree/main](https://huggingface.co/distilgpt2/tree/main) and place them in `checkpoints/distilgpt2`, e.g., `checkpoints/distilgpt2/config.json`.

To download everything, you can use `git clone https://huggingface.co/distilgpt2` (note that `git lfs install` is needed).

## Instructions   
 - The model configurations for each task can be found in its `config` directory, e.g. `task/mimic_cxr_jpg_chen/config`.
 - The jobs for a task are described in the tasks `jobs.yaml` file, e.g. `task/mimic_cxr_jpg_chen/jobs.yaml`.
 - To test the CvT2DistilGPT2 + SCST checkpoint, set `task/mimic_cxr_jpg_chen/jobs.yaml` to (default):

    ```
    cvt_21_to_distilgpt2_scst:
        train: 0
        test: 1
        debug: 0
        num_nodes: 1
        num_gpus: 1
        num_workers: 5
    ```

 - To train CvT2DistilGPT2 with teacher forcing and then test, set `task/mimic_cxr_jpg_chen/jobs.yaml` to:
 
    ```
    cvt_21_to_distilgpt2:
        train: 1
        test: 1
        debug: 0
        num_nodes: 1
        num_gpus: 1
        num_workers: 5
    ```

    or with Slurm:
 
    ```
    cvt_21_to_distilgpt2:
        train: 1
        test: 1
        debug: 0
        num_nodes: 1
        num_gpus: 1
        num_workers: 5
        resumable: 1
        sbatch: 1
        time_limit: 1-00:00:00
    ```
 - To run a job for MIMIC-CXR-JPG with Chen's labels:
    ```shell script
    python3 main.py --task mimic_cxr_jpg_chen
    ``` 
   
   or for IU X-Ray with Chen's labels:
   
    ```shell script
    python3 main.py --task iu_x_ray_chen
    ``` 

##### Note: data from the job will be saved in the `experiment` directory.

## Reference
[1] [Aaron Nicolson, Jason Dowling, and Bevan Koopman, *Improving Chest X-Ray Report Generation by Leveraging Warm-Starting*, Under review (January 2022)](https://arxiv.org/abs/2201.09405)

## Help
If you need help, please leave an issue and we will get back to you as soon as possible.


