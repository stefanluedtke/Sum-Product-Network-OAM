# SOAM: Outlying Aspect Mining via Sum-Product Networks


This repository contains the official implementation for the paper

Stefan Lüdtke, Christian Bartelt, Heiner Stuckenschmidt. Outlying Aspect Mining via Sum-Product Networks. Advances in Knowledge Discovery and Data Mining. PAKDD 2022.

For convenience, this archive includes all the required repositories and data:
* SPFlow (https://github.com/SPFlow/SPFlow), used for fitting and inference in SPNs
* The implementation of Xu et al. (2021), used for obtaining results of ATON, COIN, SiNNE on the synthetic data
* The synthetic data from www.ipd.kit.edu/mitarbeiter/muellere/HiCS
* The real-world data (and ground truth outlier explanations) from https://github.com/xuhongzuo/outlier-interpretation



## 1. SPN-based outlier explanation (SOAM)


### 1.1 Synthetic data
To computes the results for the the SPN-based outlying aspect mining method (SOAM) for synthetic data, run `python evaluate-spn-synthetic.py`. Results are stored in results/results_synthetic.csv. The script runs all datasets and hyperparameter combinations discussed in the paper. 

### 1.2 Real-World data
Run `python evaluate-spn-real.py`. Results are stored in results/results_real.csv. 

### 1.3 Using your own data
Run `main.py <your-file>.csv`. All samples are used for fitting the SPN model (i.e., your data should include outliers as well as inliers, but you do not need to provide labels). The last column of the file should be binary, indicating whether you want to compute outlying aspects for that sample (1) or not (0). Note that querying a large number of samples can take some time. The script returns a dictionary, containing the outlying aspects (=subspaces) for each queried sample, in the order in which the samples appeared in the CSV. For example, you can run `python3 main.py data/real/data/arrhythmia_pca.csv`.


## 2. State-of-the-art methods


Results for ATON, COIN and SiNNE were obtained by running the code by Xu et al. (https://github.com/xuhongzuo/outlier-interpretation). For convenience, we included their code here, including our modifications to the main scripts to run with the synthetic data. They can be found in the folder outlier-interpretation-Xu. 
CAUTION: This code requires specific python packages that are incompatible with SPFlow, thus it is recommended to create a separate environment for installing the dependencies given in outlier-interpretation-Xu/requirements.txt. Then, the code can be executed as follows:

#ATON: 
python main_synth.py --path ../data/synth-for-aton/data/  --record_name results_aton_synth --w2s_ratio auto
#COIN: change line 36 in main_synth.py to `algorithm_name = "coin"`, then run 
python main_synth.py --path ../data/synth-for-aton/data/  --record_name results_coin_synth --w2s_ratio p2n
#SiNNE:
python main2_synth.py --path ../data/synth-for-aton/data/ --ensemble_num 100 --max_level 5 --record_name results_sinne_synth

Results can be found in outier-interpretation-Xu/record.
