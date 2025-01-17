import math
import time
import copy
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DebugRF import Dataset, FairnessMetric, FairnessDebuggingUsingMachineUnlearning

'''Class for loading and preprocessing german credits dataset'''
class GermanCreditDataset(Dataset): 
    def __init__(self, rootTrain, rootTest):
        Dataset.__init__(self, rootTrain = rootTrain, rootTest = rootTest)
        self.train = self.trainDataset
        self.test = self.testDataset
        self.__renameColumnNames()
        self.trainProcessed, self.testProcessed = self.__preprocessDataset(self.train), self.__preprocessDataset(self.test)
        self.trainLattice, self.testLattice = self.__preprocessDatasetForCategorization(self.train), self.__preprocessDatasetForCategorization(self.test)
        
    def getDataset(self):
        return self.dataset, self.train, self.test

    def getDatasetWithNormalPreprocessing(self):
        return self.trainProcessed, self.testProcessed
    
    def getDatasetWithCategorizationPreprocessing(self, decodeAttributeValues = False):
        if decodeAttributeValues == True:
            return self.__decodeAttributeCodeToRealValues(self.trainLattice), self.__decodeAttributeCodeToRealValues(self.testLattice)
        return self.trainLattice, self.testLattice

    def __renameColumnNames(self):
        columns={'Column1': 'status_chec_acc', 
                'Column2': 'duration',
                'Column3': 'cred_hist',
                'Column4': 'purpose',
                'Column5': 'cred_amt',
                'Column6': 'savings',
                'Column7': 'employment',
                'Column8': 'intallment_rate',
                'Column9': 'status_and_sex',
                'Column10': 'debtors',
                'Column11': 'present_resi_since',
                'Column12': 'property',
                'Column13': 'age',
                'Column14': 'install_plans',
                'Column15': 'housing',
                'Column16': 'existing_creds',
                'Column17': 'job',
                'Column18': 'num_people_liable_to_maint',
                'Column19': 'telephone',
                'Column20': 'foreign_worker',
                'Column21': 'status'}
        self.train.rename(columns = columns, inplace = True)
        self.test.rename(columns = columns, inplace = True)
        
    def __initiateTrainTestSplit(self):
        trainX, testX, trainY, testY = train_test_split(self.dataset.drop('status', axis = 1) , 
                                                self.dataset['status'], 
                                                stratify = self.dataset['status'], 
                                                test_size=0.2)
        trainX['status'] = trainY
        testX['status'] = testY
        self.train = trainX.reset_index(drop = True)
        self.test = testX.reset_index(drop = True)

    def __preprocessDataset(self, dataset):
        df = copy.deepcopy(dataset)
        df['status_chec_acc'] = df['status_chec_acc'].map({'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}).astype(int, errors='ignore')
        df['cred_hist'] = df['cred_hist'].map({'A34': 0, 'A33': 1, 'A32': 2, 'A31': 3, 'A30': 4}).astype(int, errors='ignore')
        df['savings'] = df['savings'].map({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}).astype(int)
        df['employment'] = df['employment'].map({'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}).astype(int)    
        df['sex'] = df['status_and_sex'].map({'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}).astype(int)
        df['debtors'] = df['debtors'].map({'A101': 0, 'A102': 1, 'A103': 2}).astype(int)
        df['property'] = df['property'].map({'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}).astype(int)        
        df['install_plans'] = df['install_plans'].map({'A141': 1, 'A142': 1, 'A143': 0}).astype(int)    
        df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')],axis=1)
        df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
        df = pd.concat([df, pd.get_dummies(df['status_and_sex'], prefix='status_and_sex')],axis=1)
        df.loc[(df['cred_amt'] <= 2000), 'cred_amt'] = 0
        df.loc[(df['cred_amt'] > 2000) & (df['cred_amt'] <= 5000), 'cred_amt'] = 1
        df.loc[(df['cred_amt'] > 5000), 'cred_amt'] = 2    
        df.loc[(df['duration'] <= 12), 'duration'] = 0
        df.loc[(df['duration'] > 12) & (df['duration'] <= 24), 'duration'] = 1
        df.loc[(df['duration'] > 24) & (df['duration'] <= 36), 'duration'] = 2
        df.loc[(df['duration'] > 36), 'duration'] = 3
        df['age'] = df['age'].apply(lambda x : 1 if x > 30 else 0) # 1 if old, 0 if young
        df['job'] = df['job'].map({'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}).astype(int)    
        df['telephone'] = df['telephone'].map({'A191': 0, 'A192': 1}).astype(int)
        df['foreign_worker'] = df['foreign_worker'].map({'A201': 1, 'A202': 0}).astype(int)
        df['status'] = df['status'].map({1: 1, 2: 0}).astype(int, errors='ignore')
        df.drop(columns=['purpose', 'status_and_sex','housing'], inplace = True)
        '''Moving status column at the end'''
        cols = list(df.columns.values) 
        cols.pop(cols.index('status')) 
        df = df[cols+['status']]
        return df
    
    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'age':
                df[col] = pd.cut(df[col], 
                               [quantiles[col][0.0] - 1, 30, quantiles[col][1.0] + 1], 
                               labels = ['age = young', 'age = old'], 
                               right = True, 
                               include_lowest = True)
            elif col == 'status':
                continue
            elif col == 'cred_amt':
                df[col] = pd.cut(df[col], 
                               [0, 1365.50001, 3972.25001, 18424.0001], 
                               labels = [str(col) + ' = low', str(col) + ' = medium', str(col) + ' = high'], 
                               right = True, 
                               include_lowest = True)
            else:
                df[col] = pd.cut(df[col], 
                               [quantiles[col][0.0] - 1, quantiles[col][0.50], math.inf], 
                               labels = [str(col) + ' = low', str(col) + ' = high'], 
                               right = True, 
                               include_lowest = True)
        df['status'] = df['status'].map({1: 1, 2: 0}).astype(int, errors='ignore')
        '''Moving status column at the end'''
        cols = list(df.columns.values) 
        cols.pop(cols.index('status')) 
        df = df[cols+['status']]
        return df
    
    def __decodeAttributeCodeToRealValues(self, dataset):
        df = copy.deepcopy(dataset)
        map_code_to_real = {
            "status_chec_acc": {
                "A11": "status_chec_acc = < 0 DM",
                "A12": "status_chec_acc = 0 <=..< 200DM",
                "A13": "status_chec_acc = > =200 DM",
                "A14": "status_chec_acc = no checking account"
            },
            "cred_hist": {
                "A30": "cred_hist = no credits taken / all credits paid back duly",
                "A31": "cred_hist = all credits at this bank paid back duly",
                "A32": "cred_hist = existing credits paid back duly till now",
                "A33": "cred_hist = delay in paying off in the past",
                "A34": "cred_hist = critical account / other credits existing (not at this bank)"
            },
            "purpose": {
                "A40": "purpose = car (new)",
                "A41": "purpose = car (used)",
                "A42": "purpose = furniture/equipment",
                "A43": "purpose = radio/television",
                "A44": "purpose = domestic appliances",
                "A45": "purpose = repairs",
                "A46": "purpose = education",
                "A47": "purpose = vacation - does not exist?",
                "A48": "purpose = retraining",
                "A49": "purpose = business",
                "A410": "purpose = others"
            },
            "savings": {
                "A61": "savings = ..<100 DM",
                "A62": "savings = 100 <= .. < 500 DM",
                "A63": "savings = 500 <= .. < 1000 DM",
                "A64" : "savings = ..>=1000 DM",
                "A65": "unknown/no savings account"
            },
            "employment": {
                "A71": "employment = unemployed",
                "A72": "employment = .. < 1 year",
                "A73": "employment = 1 <= .. < 4 years",
                "A74": "employment = 4 <= .. < 7 years",
                "A75": "employment = .. >= 7 years"
            },
            "status_and_sex": {
                "A91": "status_and_sex = male: divorced/separated",
                "A92": "status_and_sex = female: divorced/separated/married",
                "A93": "status_and_sex = male: single",
                "A94": "status_and_sex = male: married/widowed",
                "A95": "status_and_sex = femaled: single"
            },
            "debtors": {
                "A101": "debtors = none",
                "A102": "debtors = co-applicant",
                "A103": "debtors = guarantor"
            },
            "property": {
                "A121": "property = real estate",
                "A122": "property = building society savings agreement/life insurance",
                "A123": "property = car or other",
                "A124": "property = unknown / no property"
            },
            "install_plans": {
                "A141": "install_plans = bank",
                "A142": "install_plans = stores",
                "A143": "install_plans = none"
            },
            "housing": {
                "A151": "housing = rent",
                "A152": "housing = own",
                "A153": "housing = for free"
            },
            "job": {
                "A171": "job = unemployed / unskilled - non-resident",
                "A172": "job = unskilled - resident",
                "A173": "job = skilled employee / official",
                "A174": "job = management / self-employed / highly qualified employee / officer" 
            },
            "telephone": {
                "A191": "telephone = none",
                "A192": "telephone = yes, registered under customer's name"
            },
            "foreign_worker": {
                "A201": "foreign_worker = yes",
                "A202": "foreign_worker = no"
            }
        }
        object_columns = [col for col in df.columns if df[col].dtypes == 'object']
        for col in object_columns:
            df[col] = df[col].map(map_code_to_real[col]).fillna(df[col])
        return df
    

myDataset = GermanCreditDataset(rootTrain = 'Dataset/german_train.csv', rootTest = 'Dataset/german_test.csv')
# print(myDataset.train)
fairnessDebug = FairnessDebuggingUsingMachineUnlearning(myDataset,
                                                        ["age", 1, 0],
                                                        "status",
                                                        FairnessMetric.SP)
print("OriginalAccuracy: " + fairnessDebug.getAccuracy() 
      + ", originalSP: " + str(fairnessDebug.getDatasetStatisticalParity()) 
      + ", originalPP: " + str(fairnessDebug.getDatasetPredictiveParity()) 
      + ", originalEO: " + str(fairnessDebug.getDatasetEqualizingOddsParity()))

# bias_inducing_subsets = fairnessDebug.latticeSearchSubsets(4, (0.05, 0.15), "normal", True)
# print(fairnessDebug.n_pruned)

df = myDataset.trainProcessed
y = df['status']
X = df.drop(['status'], axis=1)
print(len(df))

br_idx = df.loc[(df['status'] == 0) & (df['age'] == 0)].index
df = df.drop(br_idx)
df.reset_index()
X = X.drop(br_idx)
y = y.drop(br_idx)

print(len(df))
df_test = myDataset.testProcessed
y_test = df_test['status']
X_test = df_test.drop('status', axis=1)

myDataset.trainProcessed = df.copy()
fairnessDebug_drop = FairnessDebuggingUsingMachineUnlearning(myDataset,
                                                        ["age", 1, 0],
                                                        "status",
                                                        FairnessMetric.SP)
print("OriginalAccuracy: " + fairnessDebug_drop.getAccuracy() 
      + ", originalSP: " + str(fairnessDebug_drop.getDatasetStatisticalParity()) 
      + ", originalPP: " + str(fairnessDebug_drop.getDatasetPredictiveParity()) 
      + ", originalEO: " + str(fairnessDebug_drop.getDatasetEqualizingOddsParity()))


featureNames = df.columns.drop('status')
classNames = ['1', '2']

from sklearn.ensemble import RandomForestClassifier
n_estimators = 100
max_depth = None
model = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth).fit(X, y)
model.predict(X_test)

# from sklearn import tree
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=800)
# for i in range(n_estimators):
#     tree.plot_tree(model.estimators_[i],
#                feature_names = featureNames, 
#                class_names = classNames,
#                filled = True, proportion=True);
#     fig.savefig('rf_individualtree_'+str(i)+'.png')