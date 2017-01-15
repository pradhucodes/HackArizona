import pickle
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial
from collections import Counter
import re

def findNearestResume(arrResume):
    with open('resume_to_vector_.txt', 'rb') as handle:
        doc2vec = pickle.load(handle)
    model = Word2Vec.load('/Users/pradhumanswami/Desktop/w2vectors.txt')
    for resume in arrResume:
        uploadedResume = uploadedResume+resume
    uploadedResumeData = list(set(uploadedResume.lower().split()))
    resume_vector = np.zeros(100)
    for word in uploadedResumeData:
        if word in model.vocab.keys():
            resume_vector = np.add(resume_vector, model[word])
    topDicts = dict()
    for key, value in doc2vec.items():
        topDicts[key] = spatial.distance.cosine(value,resume_vector)
    sortedArr = Counter(topDicts)
    answers = []
    values = []
    email = []
    for k,v in sortedArr.most_common(10):
        answers.append(k)
        match = re.search(r'[\w\.-]+@[\w\.-]+', answers)
        email.append(match.group(0))
        values.append(v)
    return answers,values,email