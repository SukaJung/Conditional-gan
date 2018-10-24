<h1>Conditional-gan</h1>  
<h2>Code</h2>

* dataset : mnist, dog and cat dataset  
* cgan.py : mnist  
>num_latent_variable = 100
num_hidden = 128
batch_size = 64
learning_rate = 0.0001

* cgan_convolution.py : mnist with convolution layer  
>dddd

* cgan_ani.py : dog and cata dataset with convoltion layer  
>ddddd

* preprocess_data : using pretrained image segmentation (deeplabv3 tesnorflow api) to remove background except animal   
* table.pkl : filenames in dataset (pickle)

<h2>Result</h2>  

* mnist :  

![mnist image](./readme/cgan_mnist.gif)  

* dog and cat :

![dogcat image](./readme/cgan_dogcat.gif)  
