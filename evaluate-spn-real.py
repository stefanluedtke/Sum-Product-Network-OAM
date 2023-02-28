import os

import sys
sys.path.append("SPFlow/src/")

from spn.algorithms.Inference import log_likelihood, likelihood


import numpy as np
from scipy.io import arff
import pandas as pd
from itertools import product
from sklearn import metrics
from time import perf_counter

import spnOutlierDetection as spnc
import ast


import warnings

warnings.simplefilter("ignore")


def is_power_of_two(n):
    return ((n != 0) and (n & (n-1) == 0)) or n==0

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

try:
    os.remove("results/results_real.csv")
except OSError:
    pass


colns = ['index', 'dataset', 'type', 'm', 'beamwidth', 'X-threshold', 'spntype',
       'rows', 'cols', 'threshold', 'searchstrategy', 'explainstrategy', 'AUC',
       'X-Sens-iforest', 'X-Prec-iforest', 'X-Sens-copod', 'X-Prec-copod',
       'X-Sens-hbos', 'X-Prec-hbos', 'TrainTime', 'X-Time', 'i']

colns = ','.join(colns)

with open("results/results_real.csv","w") as fi:
    fi.write(colns)
    fi.write("\n")
fi.close()


params = {
"dataset": ["arrhythmia_pca","ionosphere_pca","letter_pca","optdigits_pca","pima",
            "satimage-2_pca","wbc_pca","wineQualityReds-od2","wineQualityWhites-od2"],
"type":["outlier"], 
"m": [200], 
"beamwidth":[10],
"X-threshold":[1,2.7], 
"spntype":["gaussian"], #mixed
"rows":["gmm"],
"cols":["rdc"],
"threshold":[0.3],
"searchstrategy":["beamsearch","simplify"], 
"explainstrategy":["threshold","zscore"] 
}

paramdf = expand_grid(params)


paramdf= paramdf.reset_index()

paramdf.loc[:,"AUC"] = None
paramdf.loc[:,"X-Sens-iforest"] = None
paramdf.loc[:,"X-Prec-iforest"] = None
paramdf.loc[:,"X-Sens-copod"] = None
paramdf.loc[:,"X-Prec-copod"] = None
paramdf.loc[:,"X-Sens-hbos"] = None
paramdf.loc[:,"X-Prec-hbos"] = None
paramdf.loc[:,"TrainTime"] = None
paramdf.loc[:,"X-Time"] = None



for i in range(paramdf.shape[0]):
    print(i)
    
    dataset = paramdf.loc[i,"dataset"]
    fn = "data/real/data/"+dataset+".csv"
    rows = paramdf.loc[i,"rows"]
    cols = paramdf.loc[i,"cols"]
    
    evaltype = paramdf.loc[i,"type"]
    xthreshold = paramdf.loc[i,"X-threshold"]
    spntype = paramdf.loc[i,"spntype"]
    threshold = paramdf.loc[i,"threshold"]
    
    #load data
    dat  = np.genfromtxt(fn, delimiter=',',skip_header=1)

    X = dat[:,0:(dat.shape[1]-1)]
    y = dat[:,dat.shape[1]-1]
    
    
    
    
    norms = y==0
    Xnorm = X[norms.reshape(norms.shape[0]),:]
    Xoutlier = X[np.logical_not(norms.reshape(norms.shape[0])),:]
    ntrain = np.round(Xnorm.shape[0]*0.8)
    
    Xtrain = Xnorm[0:int(ntrain),:]
    
    #use the rest of normal class plus all outliers for testing
    Xrest = Xnorm[int(ntrain):Xnorm.shape[0]]
    Xtest = np.concatenate((Xrest,Xoutlier))
    youtl = y[np.logical_not(norms.reshape(norms.shape[0]))]
    ytest = np.concatenate(([0]*Xrest.shape[0],youtl))
                           #[1]*Xoutlier.shape[0]))
                     
                            
    #run and evaluate
    m = paramdf.loc[i,"m"]
    if evaltype=="outlier":
        start_train = perf_counter()
        spn = spnc.fit_spn(X,m,rows,cols,spntype,threshold)
        end_train = perf_counter()
    else: #evaltype=="novelty"
        start_train = perf_counter()
        spn = spnc.fit_spn(Xtrain,m,rows,cols,spntype,threshold)
        end_train = perf_counter()

    #evaluate outlier detection performance of the model
    if evaltype=="outlier":
        ll = log_likelihood(spn, X)
        yytest = y==0
        auc = metrics.roc_auc_score(yytest, ll)
    else: #evaltype=="novelty
        ll = log_likelihood(spn, Xtest)
        yytest = ytest==0
        auc = metrics.roc_auc_score(yytest, ll)
    
    
    
    #evaluate explanations
    yy = y==0
    Xoutlier = X[np.logical_not(yy.reshape(yy.shape[0])),:]
    
    searchstrategy = paramdf.loc[i,"searchstrategy"]
    explainstrategy = paramdf.loc[i,"explainstrategy"]
    bw = min([paramdf.loc[i,"beamwidth"],X.shape[1]-2])
    

    if searchstrategy == "beamsearch":
        start_x = perf_counter()
        allresults = spnc.explain_beamsearch(spn, Xoutlier, maxdim=5, beamwidth=bw)
        end_x = perf_counter()
    elif searchstrategy == "simplify":
        start_x = perf_counter()
        allresults = spnc.explain_simplify(spn, Xoutlier,maxdim=5)
        end_x = perf_counter()
        

    if explainstrategy == "threshold" and searchstrategy == "beamsearch":
        explanations = spnc.extract_explanation(allresults,xthreshold)
    elif explainstrategy=="zscore" and searchstrategy == "beamsearch":
        explanations = spnc.extract_explanation_zscore(allresults,X,spn)
    elif explainstrategy == "threshold" and searchstrategy == "simplify":
        explanations = spnc.extract_explanation_simplify2(allresults,xthreshold)
    elif explainstrategy == "zscore" and searchstrategy == "simplify":
        explanations = spnc.extract_explanation_simplify4(allresults,xthreshold,X,spn)



    #load info file 1
    def eval1(ending):
        fno = "data/real/data_od_evaluation/"+dataset+ending
        dato = pd.read_csv(fno, quotechar='"', skipinitialspace=True)
        
        #convert to dict
        outlierdims = dict()
        val = 1
        for row in range(dato.shape[0]):
            val = dato.loc[row,"ano_idx"]
            x = dato.loc[row,"exp_subspace"]
            ss = ast.literal_eval(x)
            outlierdims[val] = ss

        senss = np.empty(Xoutlier.shape[0])
        precs = np.empty(Xoutlier.shape[0])
        lenss = np.empty(Xoutlier.shape[0])
        outlierrows = np.argwhere(y==1)
        outlierrows = outlierrows.reshape(outlierrows.shape[0])
        for j in range(Xoutlier.shape[0]):
           # print(i)
            truth = outlierdims[outlierrows[j]] 
            predict = explanations[j]
            truthcappredict = np.intersect1d(truth, predict)
            senss[j] = len(truthcappredict) / len(truth)
            precs[j] =  len(truthcappredict) / len(predict)
            lenss[j] = len(predict)
        return (senss,precs)
    
    senss1,precs1 = eval1("_gt_iforest.csv")
    senss2,precs2 = eval1("_gt_copod.csv")
    senss3,precs3 = eval1("_gt_hbos.csv")

    
    paramdf.loc[i,"AUC"] = auc
    paramdf.loc[i,"X-Sens-iforest"] = np.mean(senss1)
    paramdf.loc[i,"X-Prec-iforest"] = np.mean(precs1)
    paramdf.loc[i,"X-Sens-copod"] = np.mean(senss2)
    paramdf.loc[i,"X-Prec-copod"] = np.mean(precs2)
    paramdf.loc[i,"X-Sens-hbos"] = np.mean(senss3)
    paramdf.loc[i,"X-Prec-hbos"] = np.mean(precs3)
    paramdf.loc[i,"TrainTime"] = end_train - start_train
    paramdf.loc[i,"X-Time"] = end_x - start_x
    paramdf.loc[i,"i"] = i
    
    


    pd.DataFrame([paramdf.loc[i,:]]).to_csv("results/results_real.csv",index=False,mode="a",header=False)
   

