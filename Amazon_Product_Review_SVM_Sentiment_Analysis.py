import html
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV

# choose first 50 words of each review for analysis
n=50

tokenizer=TweetTokenizer()
lemmatizer=WordNetLemmatizer()

vectizer=CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    ngram_range=(1,1),  #bag bof words, #(2,2) bi-gram
    stop_words='english')
    
def tokenize(str):
    tokens=tokenizer.tokenize(str)
    return [lemmatizer.lemmatize(x) for x in tokens]

# Amazon Health and Personal Care review data can be downloaded from
# http://jmcauley.ucsd.edu/data/amazon/
# Amazon product review  csv file can be created using 'amazon product review json file loading.py'
fpath='C:/your fold/'
fl=fpath+'amz_product_review.csv'
pr=pd.read_csv(fl,usecols=['asin','overall','reviewText'])
pr.head()

#pick the first n words of each review for furthur analysis
prc=pr.copy()
selWords=lambda x: x.split()[:n]
lrWord=lambda x:x.replace('[','')
rrWord=lambda x:x.replace(']','')
cWord=lambda x:x.replace("'",'')
qWord=lambda x:x.replace(',',' ')
prc['reviewText']=prc['reviewText'].astype(str).apply(selWords)
prc['reviewText']=prc['reviewText'].astype(str).apply(lrWord)
prc['reviewText']=prc['reviewText'].astype(str).apply(rrWord)
prc['reviewText']=prc['reviewText'].astype(str).apply(cWord)
prc['reviewText']=prc['reviewText'].astype(str).apply(qWord)

#sampling
pr=pr.sample(10000)
pr['overall'].value_counts()   

# label sentiment  
prs=pr[pr['overall']!=3]
prs['sentiment']=pr['overall'].apply(lambda x:0 if x>3.0 else 1)
prs.info()

prs.describe()
prs['sentiment'].value_counts()

# prepare data for training and testing
x=prs['reviewText']
y=prs['sentiment']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
    
np.random.seed(1)
kfolds=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

svc=LinearSVC(class_weight='balanced')
svcPipeline=make_pipeline(vectizer,svc)

grid_sch=GridSearchCV(estimator=svcPipeline,
                        param_grid={'linearsvc__C':[0.01,0.1,1]},
                        cv=kfolds,
                        scoring='f1',
                        verbose=0,
                        n_jobs=-1)
                        
grid_search=grid_sch.fit(x_train,y_train)


best_score=grid_sch.best_score_
best_parameter=grid_sch.best_params_
print('best_score:',best_score)
print('best_parameter:',best_parameter)

train_pred=grid_sch.predict(x_train)
test_pred=grid_sch.predict(x_test)
print('Train:',classification_report(y_true=y_train,y_pred=train_pred))
print('Test:',classification_report(y_true=y_test,y_pred=test_pred))

