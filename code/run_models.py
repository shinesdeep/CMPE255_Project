
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
import sys
sys.path.append("../")
from preprocessing.readdata import read_csv
# from preprocessing.process_data import process_df
from models.SVM import SVMModel
from models.RandomForest import RandomForest
def main_p():
    print("Running ......") 
    X,y = read_csv.read_data()
    print("Run The Model and Predict")
    RandomForest.run_RF(X, y, 0.2)
    return

if __name__ == '__main__':
    main_p()    
