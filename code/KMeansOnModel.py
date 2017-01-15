from gensim.models import Word2Vec
import numpy as np
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

model = Word2Vec.load('/Users/pradhumanswami/Desktop/w2vectors.txt')
words_count = 0
###CODE TO CREATE SENTENCE FROM VECTORS
featureDatabase = []
featureVec = np.zeros((100,),dtype="float32")
wordset = set(model.index2word)
for text in documents:
    for t in text:
        if t in wordset:
            words_count = words_count+1
            featureVec = np.add(featureVec,model[t])
            #featureAllData.append(model[t])

    featureVec = np.divide(featureVec,words_count)
    featureDatabase.append(featureVec)
###CODE TO CREATE SENTENCE FROM VECTORS



#### K MEANS CLUSTERING ON SENTENCES OF A IMAGE DESCRIPTION #######

#print(model.syn0)
word_vectors = model.syn0
word_vectors = featureDatabase
#print(word_vectors)
#num_clusters = len(word_vectors[0])/5;


#kmeans_clustering = KMeans(n_clusters=10)

#idx = kmeans_clustering.fit_predict(word_vectors)

kmeans_model = KMeansClusterer(num_means=10,distance=cosine_distance,avoid_empty_clusters=True)
idx = kmeans_model.cluster(word_vectors,True)


#print(idx)
word_centroid_map = dict(zip(model.index2word,idx))
for cluster in range(0,10):
    wwords = []
    print("Cluster Number %d"%cluster)
    for i in range(0, len(word_centroid_map.values())):
        v = list(word_centroid_map.values())
        if (v[i] == cluster):
            temp = list(word_centroid_map.keys())
            wwords.append(temp[i])
    print(wwords)