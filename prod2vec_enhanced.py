import _pickle as cPickle
from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
import random
from scipy import spatial
import pandas as pd
df_prodattrib=pd.read_csv('users_data.csv',low_memory=False)
df=df_prodattrib[['userId','Item', 'ItemCategory', 'Sub_Category', 'Category', 'item_id','txn_datetime']]
with open('./pruned_items.pkl','rb') as f:
    data=cPickle.load(f)
print("length of data"+str(len(data)))
random.shuffle(data)
l1=int(np.floor(0.8*len(data)))
print(l1)
train_data=data[:l1]
#print(train_data)
test_data=data[l1:]
#model=Word2Vec(train_data, window=6, size=2184, workers=4, min_count=1)
print("done")
#model.save('prod2vec_modified.model')
model=Word2Vec.load('prod2vec_modified.model')
with open("modified_vectors.txt", "rb") as myFile:
          items_list= cPickle.load(myFile)
avail_items=model.wv.vocab
print(len(avail_items))
for item in avail_items:
    try:
        print(item)
        print(np.shape(items_list[float(item)]))
        model.wv.syn0[model.wv.vocab[item].index]=items_list[float(item)]
    except ValueError:
        pass
test_size = float(len(test_data))
print("test size is"+str(test_size))
hit = 0.0
j=0
for current_pattern in test_data:
    print(j)
    j=j+1
    if len(current_pattern) <10:
        test_size -= 1.0
        continue
    # Reduce the current pattern in the test set by removing the last ite
    #last_items=[]
    last_items=current_pattern.pop()
    #print(last_items)
    #last_items= [current_pattern.pop() for _ in range(0,3)]
    #print(len(last_items))
    # Keep those items in the reduced current pattern, which are also in the models vocabulary
    items = [it for it in current_pattern if it in model.wv.vocab]
    if len(items) <= 2:
        test_size -= 1.0
        continue

    # Predict the most similar items to items
    prediction = model.wv.most_similar(positive=items)
        #print("print len of prediction"+str(len(prediction)))
    # Check if the item that we have removed from the test, last_item, is among
    # the predicted ones.
    for predicted_item, score in prediction:
        if predicted_item in last_items:
            hit += 1.0
        #print last_item
        #print prediction
    #check at higher levelof item hierarchy
    '''
    for predicted_item,score  in prediction:
        try:
            #for i in last_items:
                #print(predicted_item,i)
                if df.loc[df['item_id'] == float(predicted_item), 'ItemCategory'].iloc[0] ==df.loc[df['item_id'] ==float(last_items), 'ItemCategory'].iloc[0]  :
                    hit +=1.0
        except (IndexError,ValueError):
            pass
    '''

print("hits "+str(hit))
print("test size"+str(test_size))
hit_rate=hit/test_size
print ("hit rate is"+str(hit_rate))
