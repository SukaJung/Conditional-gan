import six.moves.cPickle as Pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def loadImage(path,ARRAY=False):
    if not ARRAY:
        inImage_ = cv2.imread(path)
        inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
    else:
        inImage = path
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max
    inImage = cv2.resize(inImage,(64,64))
    return inImage

def scaling_img(img):
    img -= np.mean(img)
    img /= np.std(img)
    min_ = np.min(img)
    max_ = np.max(img)
    img -= min_
    img /= (max_-min_)
    return img

def plot(dog, cat):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(dog):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(cat):
        ax = plt.subplot(gs[10 + i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig



class DCDataset():
    def __init__(self, data_dir, index_dir):
        self.data_dir = data_dir
        with open(index_dir+'table.pkl', 'rb') as ani:
            self.table = Pickle.load(ani)


        self.cn = len(self.table)
        self.path = data_dir
        self.size = 64
        self.channel = 3

    def getbatch(self, batchsize):
        batch = []
        label = []
        for i in range(batchsize):
            r = int(np.random.randint(0, self.cn, (1,)).item())
            path1 = self.table[r]
            if "cat" in path1:
                label.append([0,1])
            else:
                label.append([1,0])
            img1 = loadImage(self.path + path1)
            batch.append(img1)
        return np.array(batch),np.array(label)