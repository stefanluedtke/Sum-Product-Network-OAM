import sys

import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli, Categorical
#from spn.io.Graphics import plot_spn
from spn.algorithms.Inference import log_likelihood, likelihood

from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.StatisticalTypes import MetaType


from sklearn import metrics

from scipy.stats import zscore



def fit_spn(Xtrain,min_instances=10,rows="kmean",cols="rdc",spntype="gaussian",threshold=0.3):
    if spntype=="gaussian":
        spn = learn_parametric(Xtrain,
                               Context(parametric_types=[Gaussian]*Xtrain.shape[1]).add_domains(Xtrain),
                               min_instances_slice=min_instances,rows=rows,cols=cols,threshold=threshold,cpus=-1)
    elif spntype=="mixed":
        ds_context = Context(meta_types=Xtrain.shape[1]*[MetaType.REAL])
        ds_context.add_domains(Xtrain)
        spn = learn_mspn(Xtrain, ds_context, min_instances_slice=min_instances,
                         rows=rows,cols=cols,threshold=threshold,cpus=-1)
    else:
        return None
    return spn



def get_discriminative_subspaces(spn,Xtest,subspaces,vectorized=False):
    if vectorized:
        return get_discriminative_subspaces_vectorized(spn,Xtest,subspaces)
    #subspaces is an array with [sample,subspacenumber,dimensionofsubspace]
    X = np.empty((Xtest.shape[0],Xtest.shape[1]))
    lls = np.empty((Xtest.shape[0],subspaces.shape[1]))
    lls[:] = np.nan
    for i in range(subspaces.shape[1]):
        subspace = subspaces[:,i,:] #a row for each sample, indicating its subspace
        X[:] = np.nan
        for j in range(Xtest.shape[0]):
            X[j,subspace[j,:]] =  Xtest[j,subspace[j,:]]
        lls[:,i] = log_likelihood(spn,X).reshape(X.shape[0])
    ranks = (lls).argsort(axis=1)
    return (ranks, lls)


def get_discriminative_subspaces_vectorized(spn,Xtest,subspaces):
    #subspaces is an array with [sample,subspacenumber,dimensionofsubspace]
    X = np.empty((Xtest.shape[0]*subspaces.shape[1],Xtest.shape[1])) 
    X[:] = np.nan
    for i in range(subspaces.shape[1]): #i-th subspace
        subspace = subspaces[:,i,:] #a row for each sample, indicating its subspace
        for j in range(Xtest.shape[0]): #j-th sample
            #print(str(i) + "_" + str(j))
            X[(i * subspaces.shape[0]) + j,subspace[j,:]] =  Xtest[j,subspace[j,:]]
            #now, X contains [[subspace0_sample0],[subspace0_sample1],...]
    lllong = log_likelihood(spn,X).reshape(X.shape[0])
    lls = np.empty((Xtest.shape[0],subspaces.shape[1]))
    for i in range(subspaces.shape[1]): #now, all results for i-th subspace
        lls[:,i] = lllong[i*subspaces.shape[0]:i * subspaces.shape[0] + Xtest.shape[0]]
    ranks = (lls).argsort(axis=1)
    return (ranks, lls)


#backward elimination search (aka simplify search)
def explain_simplify(spn,Xtest,maxdim=2,vectorized=False):
    N = Xtest.shape[1]
    allresults = dict() #for N-d dimensions to drop: 2d array with indices and
    existDims = np.empty((Xtest.shape[0],Xtest.shape[1]),dtype=int)
    for row in range(existDims.shape[0]):
        existDims[row,:] = range(existDims.shape[1])
    for d in range(1,Xtest.shape[1]): #d: how many dimensions are dropped 
        #existDims, lls = allresults[N-d]
        #create the "subspaces" array: [sample,subspacenumber,dimensionofsubspace]
        subspaces = np.empty((existDims.shape[0],existDims.shape[1],existDims.shape[1]-1),dtype=int)
        for dropi in range(existDims.shape[1]):
            ii = list(range(dropi)) + list(range(dropi+1,existDims.shape[1]))
            remDims = existDims[:,ii]
            subspaces[:,dropi,:] = remDims
        ranks,lls = get_discriminative_subspaces(spn, Xtest, subspaces,vectorized)
        #now extract the "best" subspace: for each sample, the one 
        #having the lowest ll (the one at ranks[0])
        #for each sample...
        bests = np.empty((subspaces.shape[0],subspaces.shape[2]),dtype=int)
        bestinds = ranks[:,0]
        for si in range(subspaces.shape[0]):
            bests[si,:] = subspaces[si,bestinds[si],:]
        #bests is the new existDims...
        existDims = bests
        lls.sort(axis=1)
        allresults[N-d] = (existDims,lls)
    #only keep keys <= maxdim
    for k in range(maxdim+1,max(allresults.keys())+1):
        allresults.pop(k)
    return allresults
            



#elbow (aka threshold) method for dimensionality selection (for backward search aka simplify search)
def extract_explanation_simplify2(allresults,threshold):
    dims,lls = allresults[1]
    explanations = dict()
    for j in range(dims.shape[0]): #for all samples
        dimsizes =list(allresults.keys())
        dimsizes.sort()
        dimsizes = dimsizes[::-1]
        dimsizes = dimsizes[0:(len(dimsizes)-1)]
        #do not check for smallest dimsize (it's automatically assumed to be smallest
        #dimsize if there is never a sudden increase)
        for t in dimsizes: #iterate over dimension sizes (large to small)
            dims,lls = allresults[t]
            #check whether there's a sudden increase in LL
            if lls[j,0]+threshold < lls[j,1]:
                explanations[j] = dims[j,:]
                break
        #if criterion never reached
        if j not in explanations:
            dims,lls = allresults[1]
            explanations[j] = dims[j,:]
    return explanations
         

#zscore dimensionality selection (for backward search aka simplify search)
def extract_explanation_simplify4(allresults,threshold,X,spn,doExp=False):
    dims,lls = allresults[1]
    llzs = np.empty((dims.shape[0],len(allresults.keys())))
    for t in allresults.keys(): #for all dimension sizes
        dims,lls = allresults[t]
        for j in range(dims.shape[0]):
            #for all data (X), get marginal LL in that dimension! (in dims[j,0,:])
            Xq = np.empty((X.shape[0],X.shape[1]))
            Xq[:] = np.nan
            dd = dims[j,:]
            Xq[:,dd] = X[:,dd]
            llall = log_likelihood(spn,Xq).reshape(Xq.shape[0])
            if doExp:
                llz = zscore(np.exp(np.hstack((lls[j,0],llall))))
            else:
                llz = zscore(np.hstack((lls[j,0],llall)))
            llzs[j,t-1] = llz[0]
            
    #now for each sample, threshold search
    explanations = dict()
    for j in range(dims.shape[0]): # for all samples
        for t in range(2,llzs.shape[1]):
            ll1 = llzs[j,t-1]
            ll2 = llzs[j,t-2] #smaller dimension
            if ll1+threshold < ll2:
                dims,lls = allresults[t]
                explanations[j] = dims[j,:]
                break
        #if criterion never reached
        if j not in explanations:
            #dims,lls = allresults[1]
            m = min(allresults.keys())
            dims,lls = allresults[m]
            explanations[j] = dims[j,:]
            #texp = np.argmin(llzs[j,:]) + 1
            #explanations[j] = allresults[texp][0][j,:]
    return explanations




def unique(array):
    uniq, index = np.unique(array, return_index=True,axis=0)
    return uniq[index.argsort()]


def explain_beamsearch(spn,Xtest,maxdim=2,beamwidth=10,vectorized=False):
    #initially, try all univaritate marginals
    allresults = dict()
    
    subspaces = [ [ [i] for i in range(Xtest.shape[1])] for j in range(Xtest.shape[0])]
    subspaces = np.array(subspaces)
    
    for dim in range(1,maxdim+1):
        
        ranks,lls = get_discriminative_subspaces(spn, Xtest, subspaces,vectorized)
        bestsubspaces = np.empty((subspaces.shape[0],beamwidth,subspaces.shape[2]),dtype=int)
        bestlls =  np.empty((subspaces.shape[0],beamwidth),dtype=float)
        for j in range(subspaces.shape[0]):
            #remove identical, then select top beamwidth
            subsp =  subspaces[j,ranks[j,:],:]
            uniq, index = np.unique(subsp, return_index=True,axis=0)
            subsp = uniq[index.argsort()]
            ll = lls[j,ranks[j,:]]
            lu, li = np.unique(ll, return_index=True,axis=0)
            ll = lu[li.argsort()]
            #ll = lls[j,index]
            bestsubspaces[j,:,:] = subsp[0:beamwidth,:]
            bestlls[j,:] = ll[0:beamwidth]
        
        
        allresults[dim] = (bestsubspaces.copy(),bestlls.copy())
            
        #each of those subspaces needs to be extended by one additional dimension in all possible ways
        dims = list(range(Xtest.shape[1]))
        def extend_subspace(space):
            others = filter(lambda s: s not in space,dims)
            return np.array([np.sort(np.concatenate((space,np.array([o])))) for o in others])
        def extendall(spaces):
            news = np.concatenate([extend_subspace(space) for space in spaces])
            return news
            
        newnumspaces = bestsubspaces.shape[1]*(Xtest.shape[1]-dim)
        
        newsubspaces = np.zeros((Xtest.shape[0],newnumspaces,subspaces.shape[2]+1),dtype=int)
        for i in range(Xtest.shape[0]):
            newsubspaces[i,:,:] = extendall(bestsubspaces[i,:,:])
            
        subspaces = newsubspaces

    return allresults



#elbow (aka threshold) method to get the correct dimensionality (for forward beam search)
def extract_explanation(allresults,threshold):
    dims,lls = allresults[1]
    explanations = dict()
    for j in range(dims.shape[0]): #for all samples
        for t in allresults.keys(): #for all dimension sizes
            dims,lls = allresults[t]
            dd = dims[j,:,:]
            ll = lls[j,:]
            if ll[0] < ll[1] - threshold: # the criterion (in log domain)
                explanations[j] = dd[0,:]
                break
        #if criterion never reached:
        if j not in explanations:
            t = max(allresults.keys())
            dims,lls = allresults[t]
            explanations[j] = dims[j,0,:]
    return explanations


# z score to extract explanations (for forward beam search)
def extract_explanation_zscore(allresults,X,spn,doExp=True):
    dims,lls = allresults[1]
    llzs = np.empty((dims.shape[0],len(allresults.keys())))
    for t in allresults.keys(): #for all dimension sizes
        dims,lls = allresults[t]
        for j in range(dims.shape[0]):
            #for all data (X), get marginal LL in that dimension! (in dims[j,0,:])
            Xq = np.empty((X.shape[0],X.shape[1]))
            Xq[:] = np.nan
            dd = dims[j,0,:]
            Xq[:,dd] = X[:,dd]
            llall = log_likelihood(spn,Xq).reshape(Xq.shape[0])
            if doExp:
                llz = zscore(np.exp(np.hstack((lls[j,0],llall))))
            else:
                llz = zscore(np.hstack((lls[j,0],llall)))
            llzs[j,t-1] = llz[0]
    #now for each sample, select the dims with minimum llz
    explanations = dict()
    for j in range(dims.shape[0]): # for all samples
        texp = np.argmin(llzs[j,:]) + 1
        explanations[j] = allresults[texp][0][j,0,:]
    return explanations


