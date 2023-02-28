#run outlying aspect mining for your own dataset

import argparse
import numpy as np
import pandas as pd

import sys
sys.path.append("SPFlow/src/")

import spnOutlierDetection as spnc

parser = argparse.ArgumentParser(
                    prog = 'SOAM',
                    description = 'Outlying Aspect Mining using Sum-Product Networks')

parser.add_argument('filename',help="(unlabeled) dataset as CSV file. last column indicates whether you want to get outlying aspects for the sample (1) or not (0)")   
parser.add_argument("-t","--threshold",default=2.7,help="threshold of elbow method")
parser.add_argument("-w","--width",default=10,help="beam width of search algorithm")
parser.add_argument("-s","--search",default="forward",help="search strategy (forward/backward)") #beamsearch / simplify
parser.add_argument("-e","--explanation",default="threshold",help="explanation strategy (threshold aka elbow / zscore)")        
parser.add_argument('-m',"--mparameter",default=200,help="SPN regularization parameter")
parser.add_argument("-d","--dim",default=5,help="maximum dimension of the explanation")

args = parser.parse_args()


#load data
dat  = np.genfromtxt(args.filename, delimiter=',',skip_header=1)

X = dat[:,0:(dat.shape[1]-1)]
y = dat[:,dat.shape[1]-1]
norms = y==0
Xoutlier = X[np.logical_not(norms.reshape(norms.shape[0])),:]

#fit SPN
spn = spnc.fit_spn(X,args.mparameter,"gmm","rdc","gaussian",args.threshold)

#do the search
if args.search == "forward":
    allresults = spnc.explain_beamsearch(spn, Xoutlier, maxdim=args.dim, beamwidth=args.width)
elif args.search == "backward":
    allresults = spnc.explain_simplify(spn, Xoutlier,maxdim=args.dim)


#extract the actual explanations from the search results
if args.explanation == "threshold" and args.search == "forward":
    explanations = spnc.extract_explanation(allresults,args.threshold)
elif args.explanation=="zscore" and args.search == "forward":
    explanations = spnc.extract_explanation_zscore(allresults,X,spn)
elif args.explanation == "threshold" and args.search == "backward":
    explanations = spnc.extract_explanation_simplify2(allresults,args.threshold)
elif args.explanation == "zscore" and args.search == "backward":
    explanations = spnc.extract_explanation_simplify4(allresults,args.threshold,X,spn)

#return outlying aspects (=subspaces) for each of the queried samples (which had a 1 in the last column)
print(explanations)