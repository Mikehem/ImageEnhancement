# Image Enhancement techniques

1. RetinexNet deep learning model (TF) Decomposition for Low-Light Enhancement
2. OpenCV based Decomposition for Glare Enhancement.
3. Deblur GAN (keras) implementation for Blur Enhancement.

This solution makes use of Streamlit for creating the application. You can run the solution using:
> streamlit run ImageEnhancementNetApp.py  

The solution works with the latest TF version 2.3.1, by using the compatibility option of Tensorflow. So if you are trying to run this on a lower version you might need to make changes in the code. 

Python: 3.6
tensorflow = 2.3.1 (CPU) *The script sets os.environ["CUDA_VISIBLE_DEVICES"]="-1". I have not tested the solution with GPU *
pillow
opencv
matplotlib
pylab-sdk

