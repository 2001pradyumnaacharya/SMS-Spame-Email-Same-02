import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

data = pd.read_csv(r"../spam.csv",encoding="latin-1")


data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

data.rename(columns={'v1':'target','v2':'text'},inplace=True)

encoder = LabelEncoder()

data['target'] = encoder.fit_transform(data['target'])
# print(data.head())

# print(data.isnull().sum())

data = data.drop_duplicates(keep='first')





#EDA   
print(data['target'].value_counts())
plt.pie(data['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
# plt.show()

data['NoOfChar']=data['text'].apply(len)
data['noOfWorld'] = data['text'].apply(lambda x:len(nltk.word_tokenize(x)))
data['NoOFSent'] = data['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
print(data.head())


print(data.describe())