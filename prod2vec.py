import _pickle as cPickle
from gensim.models import Word2Vec
import numpy as np
import random
import pandas as pd
df=pd.read_csv('users_data.csv',low_memory=False)
sentence=[]
with open('./pruned_items.pkl','rb') as f:
    data=cPickle.load(f)
print("length of data"+str(len(data)))
random.shuffle(data)
l1=int(np.floor(0.8*len(data)))
print(l1)
train_data=data[:l1]
#print(train_data)
test_data=data[l1:]
model=Word2Vec(train_data, window=6, size=100, workers=4, min_count=1)
# Test
test_size = float(len(test_data))
print("test size is"+str(test_size))
hit = 0.0
for current_pattern in test_data:
    if len(current_pattern) < 2:
        test_size -= 1.0
        continue
    # Reduce the current pattern in the test set by removing the last item
    #last_items=[]
    #last_items=current_pattern.pop()
    #print(last_items)
    last_item= current_pattern.pop()
    #last_items= [current_pattern.pop() for _ in range(0,3)]
    # Keep those items in the reduced current pattern, which are also in the models vocabulary
    items = [it for it in current_pattern if it in model.wv.vocab]
    if len(items) <= 2:
        test_size -= 1.0
        continue

    # Predict the most similar items to items
    prediction = model.wv.most_similar(positive=items)
    # Check if the item that we have removed from the test, last_item, is among
    # the predicted ones.
    for predicted_item, score in prediction:
            if float(predicted_item)==float(last_item):
                hit += 1.0
    # check at next level of hierarchy
    #for predicted_item,score in prediction:
    #    for i in last_items:
    #         try:
    #             if df.loc[df['item_id'] == float(predicted_item), 'Item_Category'].iloc[0] ==df.loc[df['item_id'] ==float(i), 'Item_Category'].iloc[0]:
    #                 hit +=1.0
    #         except IndexError:
    #             pass


print("hits "+str(hit))
print("test size"+str(test_size))
hit_rate=hit/test_size
print ("hit rate is"+str(hit_rate))
