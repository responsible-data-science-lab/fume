# FUME
FUME is a system that explains a fairness violation by leveraging machine unlearning techniques. We demonstrate FUME's effectiveness on random forest classifiers where explanations are generated in the form of patterns or subsets of data points in its training dataset that can be attributed the most toward bias exhibited by the random forest. FUME leverages the unlearning capabilities of the data removal-enabled version of a random forest (DaRE-RF - https://github.com/jjbrophy47/dare_rf) and employs a hierarchically ordered lattice tree structure to prune the huge search space of possible training data subsets.

# Main Algorithm
The main algorithm is written in fume.ipynb. It contains 2 classes - "Dataset" and "FairnessDebuggingUsingMachineUnlearning".
1. "Dataset" class is a template class for preprocessing dataset. Anyone wanting to use FUME have to preprocess their dataset by inheriting this class and providing definitions to their template functions. Dataset needs to be preprocess to get 2 datasets - a numerical version and a purely categorical version. Numerical attributes can be converted to categorical using bucketization as per domain knowledge. Care should be taken that after both these preprocessing the number of rows in both versions of dataset is same in both training and testing. The number of columns can be different. Also indices of data intances in both versions should match.
2. "FairnessDebuggingUsingMachineUnlearning" is the main class where all the logic of FUME exists. To get the explanations for fairness-based bias for you random forest model, simply call the create an instance of this call by providing it with an instance of "Dataset" class and few other arguments. The main function to call to get explanations after this is - "latticeSearchSubsets()". All the hyperparameters of the algorithm are passed as arguments to this function.

# How Good are the Explanations?
After obtaining get the most bias-inducing subsets, one can further analyze them by considering the positive label proportion for sensitive groups and feature importance deviations after deleting this subset from the model's training. The 2 functions which do this are - "drawInferencesFromResultSubsets()" and "getFeatureImportanceChanges()" respectively. 

# Datasets
We have provided FUME's implementation on 5 datasets - Adult Income, German Credit, Stop Question Frisk, MEPS and ACS Income dataset. You can check out FUME's workings on these datasets by simply running the 5 files provided for their fairness debugging. Feel free to tinker around with hyperparameters and/or preprocessing to check their impact.

# Testing of FUME
1. "Testing_DaRE_RF_Effect_on_Fairness.ipynb" --> Testing DaRE-RF unlearning technique's effect on fairness metric.
2. "Testing_FUME_Efficiency_for_Dimensions_Variation.ipynb" --> Testing FUME's efficiency in terms of time as dataset dimensions (num_rows x num_cols) increases.
3. "Testing_FUME_Efficiency_for_Instances_and_Features_Variation.ipynb" --> Testing FUME's efficiency in terms of time as number of dataset instances and features (along with feature cardinality) increases.
   
