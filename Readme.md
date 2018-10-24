<h1>Conditional-gan</h1>  
<h2>Code</h2>

* dataset : mnist, dog and cat dataset  
* cgan.py : mnist  
>__num_latent_variable__ = 100  
>__num_hidden__ = 128  
>__batch_size__ = 64  
>__learning_rate__ = 0.0001  

* cgan_convolution.py : mnist with convolution layer  
>__generator__ = 2 fc(fully connected layer), 2 convolution_transpose layers  
__discriminator__ = 2 convolution layers and 2 fc(fully connected layer)  
__ngf,ndf__ = 64  
__num_latent_variable__ = 100  
__batch_size__ = 128  
__learning_rate__ = 0.0002  

* cgan_ani.py : dog and cata dataset with convoltion layer  
>ddddd

* preprocess_data : using pretrained image segmentation (deeplabv3 tesnorflow api) to remove background except animal   
* table.pkl : filenames in dataset (pickle)

<h2>Result</h2>  

* mnist :  

![mnist image](./readme/cgan_mnist.gif)  

* dog and cat :

![dogcat image](./readme/cgan_dogcat.gif)  
