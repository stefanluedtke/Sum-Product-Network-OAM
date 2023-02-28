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

import warnings

warnings.simplefilter("ignore")


def is_power_of_two(n):
    return ((n != 0) and (n & (n-1) == 0)) or n==0

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

try:
    os.remove("results/results_synthetic.csv")
except OSError:
    pass


colns = ['index', 'nvar', 'datanum', 'type', 'm', 'beamwidth', 'X-threshold',
       'spntype', 'rows', 'cols', 'threshold', 'searchstrategy',
       'explainstrategy', 'AUC', 'X-Sens', 'X-Prec', 'TrainTime', 'QueryTime',
       'X-Time', 'Extract-Time', 'i']

colns = ','.join(colns)

with open("results/results_synthetic.csv","w") as fi:
    fi.write(colns)
    fi.write("\n")
fi.close()

params = {
"nvar":["010","020","030","040","050","075","100"],
"datanum":[0,1,2],
"type":["outlier"],
"m": [200],  
"beamwidth":[10], 
"X-threshold":[1,2.7], 
"spntype":["gaussian"],
"rows":["gmm"],
"cols":["rdc"],
"threshold":[0.3],
"searchstrategy":["beamsearch","simplify"],
"explainstrategy":["threshold","zscore"]
}

paramdf = expand_grid(params)


paramdf= paramdf.reset_index()

paramdf.loc[:,"AUC"] = None
paramdf.loc[:,"X-Sens"] = None 
paramdf.loc[:,"X-Prec"] = None
paramdf.loc[:,"TrainTime"] = None
paramdf.loc[:,"QueryTime"] = None
paramdf.loc[:,"X-Time"] = None
paramdf.loc[:,"Extract-Time"] = None


for i in range(paramdf.shape[0]):
    print(i)
    
    nvar = paramdf.loc[i,"nvar"]
    datanum = paramdf.loc[i,"datanum"]
    fn = "synth_multidim_"+nvar+"_00"+str(datanum)
    rows = paramdf.loc[i,"rows"]
    cols = paramdf.loc[i,"cols"]
    
    evaltype = paramdf.loc[i,"type"]
    xthreshold = paramdf.loc[i,"X-threshold"]
    spntype = paramdf.loc[i,"spntype"]
    threshold = paramdf.loc[i,"threshold"]
    
    #load data
    datfile = "data/hics-synth/"+fn+".arff"
    dataset,f = arff.loadarff(datfile)

    dat = np.array(dataset.tolist(),dtype=float)
    
    #filter out samples with more than one outlier subspace:
    good = list(map(is_power_of_two,dat[:,(dat.shape[1]-1)].astype(int)))
    dat = dat[good,:]
    
    y = dat[:,dat.shape[1]-1]
    X = dat[:,0:(dat.shape[1]-1)]
    
    
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
                          
    #load info file
    infofile =  "data/hics-synth/"+fn+".info"
    file = open(infofile)
    for j in range(4):
        file.readline()

    #rows = []
    outlierdims = dict()
    val = 1
    for row in file:
        if row =="\n":
            break
        ss = " ".join(row.split()).split("]")[0].split("[")[1].split(",")
        ss = list(map(int,ss))
        outlierdims[val] = ss
        val = val * 2
    file.close()
    
    
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
        start_query =  perf_counter()
        ll = log_likelihood(spn, X)
        end_query =  perf_counter()
        yytest = y==0
        auc = metrics.roc_auc_score(yytest, ll)
    else: #evaltype=="novelty
        start_query =  perf_counter()
        ll = log_likelihood(spn, Xtest)
        end_query =  perf_counter()
        yytest = ytest==0
        auc = metrics.roc_auc_score(yytest, ll)


    #evaluate explanations
    yy = y==0
    Xoutlier = X[np.logical_not(yy.reshape(yy.shape[0])),:]
    youtlier = y[np.logical_not(yy.reshape(yy.shape[0]))]


    searchstrategy = paramdf.loc[i,"searchstrategy"]
    explainstrategy = paramdf.loc[i,"explainstrategy"]
    bw = min([paramdf.loc[i,"beamwidth"],X.shape[1]])
    
    if searchstrategy == "beamsearch":
        start_x = perf_counter()
        allresults = spnc.explain_beamsearch(spn, Xoutlier, maxdim=5, beamwidth=bw,vectorized=True)
        end_x = perf_counter()
    elif searchstrategy == "simplify":
        start_x = perf_counter()
        allresults = spnc.explain_simplify(spn, Xoutlier,maxdim=5,vectorized=True)
        end_x = perf_counter()
        
    start_extract = perf_counter()
    if explainstrategy == "threshold" and searchstrategy == "beamsearch":
        explanations = spnc.extract_explanation(allresults,xthreshold)
    elif explainstrategy=="zscore" and searchstrategy == "beamsearch":
        explanations = spnc.extract_explanation_zscore(allresults,X,spn)
    elif explainstrategy == "threshold" and searchstrategy == "simplify":
        explanations = spnc.extract_explanation_simplify2(allresults,xthreshold)
    elif explainstrategy == "zscore" and searchstrategy == "simplify":
        explanations = spnc.extract_explanation_simplify4(allresults,xthreshold,X,spn)
    end_extract = perf_counter()



    senss = np.empty(Xoutlier.shape[0])
    precs = np.empty(Xoutlier.shape[0])
    lenss = np.empty(Xoutlier.shape[0])
    for j in range(Xoutlier.shape[0]):
        truth = outlierdims[youtlier[j]]
        predict = explanations[j]
        truthcappredict = np.intersect1d(truth, predict)
        senss[j] = len(truthcappredict) / len(truth)
        precs[j] =  len(truthcappredict) / len(predict)
        lenss[j] = len(predict)
        

    
    paramdf.loc[i,"AUC"] = auc
    paramdf.loc[i,"X-Sens"] = np.mean(senss)
    paramdf.loc[i,"X-Prec"] = np.mean(precs)
    paramdf.loc[i,"TrainTime"] = end_train - start_train
    paramdf.loc[i,"QueryTime"] = end_query - start_query
    paramdf.loc[i,"X-Time"] = end_x - start_x
    paramdf.loc[i,"Extract-Time"] = end_extract - start_extract
    paramdf.loc[i,"i"] = i


    pd.DataFrame([paramdf.loc[i,:]]).to_csv("results/results_synthetic.csv",index=False,mode="a",header=False)
   

    
    
    
