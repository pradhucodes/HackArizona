

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle

with open('resume_to_vector_.txt', 'rb') as handle:
    doc2vec = pickle.load(handle)

clusters = MiniBatchKMeans(n_clusters=200, max_iter=10,batch_size=200,
                        n_init=1,init_size=2000)
X = np.array([i.T for i in doc2vec.values()])
y = [i for i in doc2vec.keys()]
clusters.fit(X)
from collections import defaultdict
cluster_dict=defaultdict(list)
for word,label in zip(y,clusters.labels_):
    cluster_dict[label].append(word)




