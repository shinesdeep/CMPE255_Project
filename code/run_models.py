
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
from SVM import SVMModel
from NaiveBayes import NBModel 
import sys
sys.path.append("../")
from preprocessing_scripts.readdata import read_csv
from preprocessing_scripts.process_data import process_df

def main_p():
    print("Running ......")
    
    homecredit = read_csv.read_data()
    proc_data = process_df.preprocess(firedata)

    print("Run The Model and Predict")
    X = proc_data['COLUMNS']
    y = proc_data['COLUMN']
    NBModel.run_NB(X, y, 0.2)
    DecTreeModel.run_tree(X,y,0.2)
    
    return



if __name__ == '__main__':
    main_p()    
