# Front-Door Adjustment with Text Prompts: A Causal Approach to Image Domain Generalization

## Installation
To install conda env with conda, run the following command in your terminal:
```sh
conda env create -n ViT_DGbed --file ViT_DGbed.yml
```
Activate the conda environment:
```sh
conda activate ViT_DGbed
```
## Datasets

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data --dataset pacs
```
Note: for downloading other datasets change --dataset pacs with other datasets (e.g., vlcs, office_home).

## Training CI-TP

```sh
bash ./CI-TP_pacs.sh
bash ./CI-TP_oh.sh
bash ./CI-TP_vlcs.sh
```