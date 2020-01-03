from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load('Word2Vec_1.model') #load model

#print(model.wv.vocab.keys())
#print(model.wv.similar_by_word("เอเลี่ยน"))

max_value = 0
min_value = 0
for i in model.wv.vocab.keys():
    max_value = max(np.max(model.wv.get_vector(i)),max_value)
    min_value = min(np.min(model.wv.get_vector(i)),min_value)
    print(min_value,"-",max_value)