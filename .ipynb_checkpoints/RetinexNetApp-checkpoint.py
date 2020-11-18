from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
from matplotlib import pyplot as plt
sys.path.append(BASE_PATH)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#-----------------------------------------
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
#-----------------------------------------
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from model import lowlight_enhance
from utils import *


from pylab import *
import cv2
#-----------------------------------------
from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image
#-----------------------------------------
dblur_weight_path = os.path.join(BASE_PATH, "model/deblurGan", "generator.h5")

def lowlight_test(lowlight_enhance, img):
    test_low_data = []
    test_high_data = []
    # Load the image to process
    test_low_data.append(img)
    # 
    R_low, I_low, I_delta, S = lowlight_enhance.test(test_low_data, test_high_data)
    return  R_low, I_low, I_delta, S

def Retinex_Decomposition():
    st.subheader("Deep Retinex Decomposition for Low-Light Enhancement")
    with tf.Session() as sess:
        model = lowlight_enhance(sess)
        # get the image
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg"]) #"png", "jpg", "jpeg",
        if img_file_buffer is not None:
            sample_image = Image.open(img_file_buffer)
            sample_image = np.array(sample_image, dtype="float32") / 255.0
            #st.image(sample_image)
            R_low, I_low, I_delta, S = lowlight_test(model, sample_image)
            S_im = np.squeeze(S[0])
            S_im = Image.fromarray(np.clip(S_im * 255.0, 0, 255.0).astype('uint8'))
            with st.spinner('Enhancing Image...'):
                # Display
                plt.figure(figsize=(25, 20))
                plt.subplot(121)
                plt.title('Original Image')
                plt.axis('off')
                plt.imshow(sample_image)

                plt.subplot(122)
                plt.title('Final Image')
                plt.axis('off')
                plt.imshow(S_im)
                st.pyplot()
                time.sleep(10)
            
def OpenCV_Glare_Decomposition():
    st.subheader("OpenCV based Glare removal")
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg"]) #"png", "jpg", "jpeg",
    if img_file_buffer is not None:
        sample_image = Image.open(img_file_buffer)
        image_in = np.array(sample_image)
        #image_in = cv2.cvtColor(np.array(sample_image), cv2.COLOR_RGB2BGR) 
        # Convert RGB to BGR 
        #image_in = image_in[:, :, ::-1].copy() 
        # split into HSV components
        h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        # Find all pixels that are not very saturated
        saturation_threshold = st.sidebar.slider("Saturation Threshold", min_value=100, max_value=300, value=180, 
                                                        step=10, format=None, key=None) #180
        nonSat = s < saturation_threshold
        # Slightly decrease the area of the non-satuared pixels by a erosion operation.
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)

        # Set all brightness values, where the pixels are still saturated to 0.
        v2 = v.copy()
        v2[nonSat == 0] = 0;

        glare_threshold = st.sidebar.slider("Glare Threshold", min_value=100, max_value=400, value=200, 
                                                   step=10, format=None, key=None) #200
        glare = v2 > glare_threshold;    # filter out very bright pixels.
        # Slightly increase the area for each pixel
        glare = cv2.dilate(glare.astype(np.uint8), disk);  
        glare = cv2.dilate(glare.astype(np.uint8), disk);
        corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        with st.spinner('Enhancing Image...'):
            # Display
            plt.figure(figsize=(25, 20))
            plt.subplot(121)
            plt.title('Original Image')
            plt.axis('off')
            plt.imshow(image_in)

            plt.subplot(122)
            plt.title('Final Image')
            plt.axis('off')
            plt.imshow(corrected)
            st.pyplot()
            time.sleep(10)

def DeBlur_GAN_Decomposition():
    st.subheader("GAN based Blur removal")
    g = generator_model()
    g.load_weights(dblur_weight_path)
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg"]) #"png", "jpg", "jpeg",
    if img_file_buffer is not None:
        sample_image = Image.open(img_file_buffer)
        image_resize = sample_image.resize((256,256))
        #st.write(type(image_resize), image_resize.size)
        image_in = np.array([preprocess_image(image_resize)])
        #st.write(image_in.shape)
        x_test = image_in
        generated_images = g.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)
        #st.write("generated images size:", generated_images.shape)
        for i in range(generated_images.shape[0]):
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            #output = np.concatenate((x, img), axis=1)
            im = Image.fromarray(img.astype(np.uint8))
            
        with st.spinner('Enhancing Image...'):
            # Display
            plt.figure(figsize=(25, 20))
            plt.subplot(121)
            plt.title('Original Image')
            plt.axis('off')
            plt.imshow(image_resize)

            plt.subplot(122)
            plt.title('Final Image')
            plt.axis('off')
            plt.imshow(im)
            st.pyplot()
            time.sleep(10)
                
def main():
    st.title("Image pre-processing for image enhancement")
    #
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    st.sidebar.title("Which Pre Processing Model to run")
    app_mode = st.sidebar.selectbox("Choose the model mode",
        ["Image Enhancement & Preprocessing", "Retinex Decomposition", "OpenCV Glare Removal", "DL Blur Removal"])
    if app_mode == "Image Enhancement & Preprocessing":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Retinex Decomposition":
        result_load = Retinex_Decomposition()
    elif app_mode == "OpenCV Glare Removal":
        #st.sidebar.warning("Resnet model functionality still in progress.")
        result_load = OpenCV_Glare_Decomposition()
    elif app_mode == "DL Blur Removal":
        #st.sidebar.warning("Resnet model functionality still in progress.")
        result_load = DeBlur_GAN_Decomposition()
    else:
        st.sidebar.success('To continue select the "Model".')
        st.markdown(intro_markdown, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 
