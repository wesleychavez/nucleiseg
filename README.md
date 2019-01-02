# nucleiseg: Finding Nuclei in Divergent Images
This implementation trains U-Net on images of segmented nuclei for kaggle's 2018 Data Science Bowl.  It performs a grid search on batch sizes and optimizers, and k-fold cross-validation on the dataset.  Metrics plots are saved for each hyperparameter combination.
## Training U-Net
```shell
python train.py
```
