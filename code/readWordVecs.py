from gensim.models import Word2Vec
model = Word2Vec.load('C:\Users\Arjun\Desktop\w2vectors')

print(model.most_similar('python'))
print(model.most_similar('jquery'))
print(model.similarity('developer', 'manager'))



import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

#the vector dictionary of the model
word2vec_dict={}
for i in model.vocab.keys():
    try:
        word2vec_dict[i]=model[i]
    except:
        pass

#This is also interesting to try with Ward Hierarchical Clustering
clusters = MiniBatchKMeans(n_clusters=200, max_iter=10,batch_size=200,
                        n_init=1,init_size=2000)
X = np.array([i.T for i in word2vec_dict.itervalues()])
y = [i for i in word2vec_dict.iterkeys()]
clusters.fit(X)
from collections import defaultdict
cluster_dict=defaultdict(list)
for word,label in zip(y,clusters.labels_):
    cluster_dict[label].append(word)

for i in range(0,5):
        print('Printing cluster '+str(i))
        #cluster_dict[i].sort()
        print(cluster_dict[i])

for i in range(len(cluster_dict)):
    if 'java' in cluster_dict[i]:
        cluster_dict[i].sort()
        print(cluster_dict[i])


