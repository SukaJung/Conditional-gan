import six.moves.cPickle as Pickle
import os

dataset_dir = '/home/suka/dataset/preprocessed_catdog/'
ani = []

print("start")

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        ani.append(filename)
        
with open('table.pkl', 'wb') as table:
    Pickle.dump(ani, table)
    
print('done')