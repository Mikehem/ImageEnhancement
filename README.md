# Image Enhancement techniques

1. RetinexNet deep learning model (TF) Decomposition for Low-Light Enhancement - (project)[https://arxiv.org/abs/1808.04560] (Project Page)[https://daooshee.github.io/BMVC2018website/]
2. OpenCV based Decomposition for Glare Enhancement.

> The basic procedure consists of 3 steps:  
>1. Decompose the original image into a color, saturation and brightness component.
>2. Find particularly bright areas in the image.
>3. Inpaint these ares with the surrounding pixels.

3. Deblur GAN (keras) implementation for Blur Enhancement. (article)[https://www.sicara.ai/blog/2018-03-20-GAN-with-Keras-application-to-image-deblurring]

This solution makes use of Streamlit for creating the application. You can run the solution using:
> streamlit run ImageEnhancementNetApp.py  

The solution works with the latest TF version 2.3.1, by using the compatibility option of Tensorflow. So if you are trying to run this on a lower version you might need to make changes in the code. 
*The script sets os.environ["CUDA_VISIBLE_DEVICES"]="-1". I have not tested the solution with GPU*  
 
Python: 3.6  
tensorflow = 2.3.1 (CPU)  
pillow  
opencv  
matplotlib  
pylab-sdk  

