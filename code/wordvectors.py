import nltk
from gensim.models import Word2Vec
from gensim import models
import numpy as np
import networkx as ngraphx
from nltk.cluster.util  import cosine_distance
from nltk.cluster.kmeans  import KMeansClusterer
from collections import defaultdict
import os

######CODE TO GET A DESCRIPTIN #####
import json

from sklearn.cluster import KMeans

stoplist = set('for a of the and to in'.split())

######CODE TO GET A DESCRIPTOIN #####
data = []
json_data = ''
i = 0
path = '/Users/pradhumanswami/Desktop/resumes'

for filenames in os.listdir(path):
    with open(os.path.join(path, filenames)) as myfile:
        data.append(json.loads(myfile.read()))
        i = i+1
length = len(data)
######CODE TO GET A DESCRIPTOIN #####

documents = []
frequency = defaultdict(int)
for givenresumefiles in data:
    length = len(givenresumefiles)
    for id in range(0,length):
        resume_data = givenresumefiles[id]["resume_text"]
        documents.append(resume_data)

frequency = defaultdict(int)
texts = [[word for word in document.lower().split() if word not in stoplist]for document in documents]
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]
#print(texts)
documents = texts
#REMOVED LOW FREQUENCY DATA

###LOAD VECTOR MODEL
model = Word2Vec(documents, size=100, window=12, min_count=5, workers=4)
model.save('/Users/pradhumanswami/Desktop/w2vectors.txt')
wordVect = defaultdict()
#for word in documents:
#    wordVect[word] =