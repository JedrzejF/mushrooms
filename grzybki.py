import sys
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv('train/train.tsv', sep='\t', header=None)
dev_data = pd.read_csv('dev-0/in.tsv', sep='\t', header=None)
test_data = pd.read_csv('test-A/in.tsv', sep='\t', header=None)

train_data = train_data.drop(train_data.columns[[1, 10, 15, 16]], axis=1)
dev_data = dev_data.drop(dev_data.columns[[0, 9, 14, 15]], axis=1)
test_data = test_data.drop(test_data.columns[[0, 9, 14, 15]], axis=1)

dev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', header=None)

colnames = ['ep','csurface','ccolor','bruises','odor','gattach','gspacing','gzise','gcolor',
            'sroot','ssaring','ssbring','scaring','vcolor','rnumber','rtype','spcolor','popul','hab']
train_data.columns = colnames

lb = LabelEncoder()

for i in train_data:
    train_data[i] = lb.fit_transform(train_data[i])
for i in dev_data:
    dev_data[i] = lb.fit_transform(dev_data[i])
for i in test_data:
    test_data[i] = lb.fit_transform(test_data[i])
    
train_data.to_csv('label.csv')

train_features = train_data.loc[: , 'csurface':'hab']
train_result = train_data.loc[: , 'ep']

model = GaussianNB()
model = model.fit(train_features, train_result)

dev_output = model.predict(dev_data)
dev_output = ['e' if x==0 else 'p' for x in dev_output]
test_output = model.predict(test_data)
test_output = ['e' if x==0 else 'p' for x in test_output]

# Counting percentage of correct predictionsfor dev-0
dev_expected = dev_expected[0].values.tolist()
dev_evaluation = pd.DataFrame(np.column_stack([dev_output, dev_expected]), 
                               columns=['predicted', 'actual'])
good = 0
for index, row in dev_evaluation.iterrows():
    if row['predicted'] == row['actual']:
        good+=1
print(good/len(dev_evaluation))

# Saving files
with open('./test-A/out.tsv', 'w') as output:
  for r in test_output:
      output.write(str(r) + '\n')
        
with open('./dev-0/out.tsv', 'w') as output:
  for r in dev_output:
      output.write(str(r) + '\n')
