# Large Chemical Language Models for Property Prediction and High-throughput Screening of Ionic Liquids

This is the official implementation of "Large Chemical Language Models for Property Prediction and High-throughput Screening of Ionic Liquids". In this study, we introduce ILBERT, a large-scale chemical language model designed to predict twelve key physicochemical and thermodynamic properties of ionic liquids (ILs). To facilitate the widespread use of ILBERT for assisting researchers in designing ILs for specific processes, a web server thereon is developed on the http://ai4solvents.com/prediction.

# How to get started

## 1. Setting Up the ILBERT Environment

```bash
conda create --name ilbert python=3.11
conda activate ilbert
pip install -r requirements.txt
```

## 2. Download the database

Due to the large size of the pre-training dataset and the virtual screening dataset, we provide these two datasets in [Zenodo](https://zenodo.org/records/14601320?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRlNDY4MGUyLTYzZjgtNDg4Ny1iODJiLWVjZmQxYzZjMGMzNyIsImRhdGEiOnt9LCJyYW5kb20iOiIxNDM2Y2Y1Mjg4YjU4ZmQzZTRiMTkyMDYyYTkzZWRhZSJ9.8Oj2fGQBaMM-cxgO-PVH8qJZrKh4d5ySpacbqo_q03S48P8wswvHOulyWddIyv9sfxeq9uyOoatJEcykmFs3JA). The pre-training dataset contains 31M unlabeled IL-like molecules, and the virtual screening dataset contains 8,333,096 synthetically-feasible ILs including the 12 key properties we predict.

## 3. Fine-tuning

To fine-tune the ILBERT on other downstream IL properties datasets, please run `ILBERT/finetune.py` and change the path and target. Pre-trained model and all model to predict properties of ILs can be found in [Zenodo](https://zenodo.org/records/14601320?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRlNDY4MGUyLTYzZjgtNDg4Ny1iODJiLWVjZmQxYzZjMGMzNyIsImRhdGEiOnt9LCJyYW5kb20iOiIxNDM2Y2Y1Mjg4YjU4ZmQzZTRiMTkyMDYyYTkzZWRhZSJ9.8Oj2fGQBaMM-cxgO-PVH8qJZrKh4d5ySpacbqo_q03S48P8wswvHOulyWddIyv9sfxeq9uyOoatJEcykmFs3JA). More IL properties will be added in the future.

## 4. Screening

Please download the new predicted database from [Zenodo](https://zenodo.org/records/15046370). Run the jupyter notebook `ILBERT/screening.ipynb` to get the screening results.

