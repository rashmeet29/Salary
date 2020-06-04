import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_excel('Train Dataset.xlsx')

D=dataset.drop(['Sr no','job_description','job_type','City coded','company_name_encoded'],axis=1)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0,19802):
    job_desig = re.sub('[^a-zA-Z]', ' ', D['job_desig'][i])
    job_desig = job_desig.lower()
    job_desig = job_desig.split()
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    job_desig = [ps.stem(word) for word in job_desig if not word in set(stopwords.words('english'))]
    job_desig = ' '.join(job_desig)
    corpus.append(job_desig)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

corpus1= []
for i in range(0, 19802):
    key_skills = re.sub('[^a-zA-Z]', ' ', str(D['key_skills'][i]))
    key_skills = key_skills.lower()
    key_skills = key_skills.split()
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    key_skills = [ps.stem(word) for word in key_skills if not word in set(stopwords.words('english'))]
    key_skills = ' '.join(key_skills)
    corpus1.append(key_skills)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X1 = cv.fit_transform(corpus1).toarray()

X=pd.DataFrame(X)
X1=pd.DataFrame(X1)





