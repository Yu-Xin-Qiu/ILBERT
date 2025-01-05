# Large Chemical Language Models for Property Prediction of Ionic Liquids

This is the official implementation of "Large Chemical Language Models for Property Prediction of Ionic Liquids". In this study, we introduce ILBERT, a large-scale chemical language model designed to predict twelve key physicochemical and thermodynamic properties of ionic liquids (ILs). To facilitate the widespread use of ILBERT for assisting researchers in designing ILs for specific processes, a web server thereon is developed on the http://ai4solvents.com/prediction.

# How to get started

## 1. Setting Up the ILBERT Environment

```bash
conda create --name ilbert python=3.11
conda activate ilbert
pip install -r requirements.txt
```

## 2. Download the database

Due to the large size of the pre-training dataset and the virtual screening dataset, we provide these two datasets in https://zenodo.org/uploads/14601320. The pre-training dataset contains 31M unlabeled IL-like molecules, and the virtual screening dataset contains 8,333,096 synthetically-feasible ILs including the 12 key properties we predict.

## 3. Fine-tuning

To fine-tune the ILBERT on other downstream IL properties datasets, run `ILBERT/CV.py` and change the path. Pre-trained model can be found in `ILBERT/pretrained_model.pth`. More IL properties will be added in the future.

## 4. Screening

Run the jupyter notebook `ILBERT/screening.ipynb` to get the screening results.

