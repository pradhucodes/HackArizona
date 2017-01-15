import nltk
from gensim.models import Word2Vec
from gensim import models
import numpy as np
import networkx as ngraphx
from nltk.cluster.util  import cosine_distance
from nltk.cluster.kmeans  import KMeansClusterer
from collections import defaultdict
import os
import time
######CODE TO GET A DESCRIPTIN #####
import json

from sklearn.cluster import KMeans

stoplist = set('for a of the and to in'.split())

######CODE TO GET A DESCRIPTOIN #####
data = []
json_data = ''
i = 0
path = '/Users/pradhumanswami/Downloads/resumes'

for filenames in os.listdir(path):
    with open(os.path.join(path, filenames)) as myfile:
        data.append(json.loads(myfile.read()))
        i = i+1
length = len(data)
######CODE TO GET A DESCRIPTOIN #####

from gensim.models import Word2Vec
model = Word2Vec.load('/Users/pradhumanswami/Desktop/w2vectors.txt')

resume_to_vector = defaultdict()
import pickle
documents = []
frequency = defaultdict(int)
for givenresumefiles in data:
    length = len(givenresumefiles)
    for id in range(0,length):
        start = time.clock()
        resume_data = givenresumefiles[id]["resume_text"]
        if(len(resume_data)<30):
            continue
        resume_words = list(set(resume_data.lower().split()))
        resume_vector = np.zeros(100)
        freq = dict()
        for word in resume_words:
            try:
                if (freq[word] > 20):
                    break
                freq[word] = freq[word] + 1
            except:
                freq[word] = 0
            if word in model.vocab.keys():
                resume_vector = np.add(resume_vector, model[word])

        resume_to_vector[resume_data] = resume_vector
        end = time.clock()
        print(end - start)
out_filename = 'resume_to_vector_.txt'
output = open(out_filename, 'ab+')
pickle.dump(resume_to_vector, output)
output.close()
