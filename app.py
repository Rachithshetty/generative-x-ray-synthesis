import streamlit as st
import os
import time
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import webbrowser
from utils import fetch_disease_info, load_and_test_model, generate_images, classify_images, save_to_hdf5

def start_tensorboard(logdir):
    # Start TensorBoard subprocess
    tensorboard_cmd = f"tensorboard --logdir={logdir}"
    subprocess.Popen(tensorboard_cmd, shell=True)

st.set_page_config(layout='wide')
st.title('Generative X-ray Synthesis For Enhanced Disease Prediction In Medical Imaging')
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 250px;
        overflow: hidden;} /* Hide overflow to prevent vertical scrolling */
    """,
    unsafe_allow_html=True
)

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
          'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
          'Pneumonia', 'Pneumothorax']
with h5py.File('chest_xray.h5', 'r') as h5_data:
    real_images = h5_data['images'][:]  # Load all images
    real_labels = {label: h5_data[label][:] for label in labels}

st.subheader('Chest X-ray Images')

# Dropdown menu to select label
selected_label = st.selectbox('Select Label', labels)
col1, col2 = st.columns(2)
with col1:
    label_indices = np.where(real_labels[selected_label] == 1)[0]
    label_index = random.choice(label_indices)

    # Plot the image
    fig, ax = plt.subplots()
    selected_image = real_images[label_index]
    ax.imshow(selected_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.text(selected_label+':')
    disease_info = fetch_disease_info(selected_label)
    if disease_info:
        st.write(disease_info)
    else:
        st.write("No information found.")
    
st.subheader('Test the Prediction Model')
st.text("Loading model...")
decoded_predictions, raw_predictions, accuracy = load_and_test_model(selected_image, selected_label)
st.text(f"Predicted Raw values: {raw_predictions}")
st.text(f"Decoded Predictions: {decoded_predictions}")
st.text(f"Accuracy: {accuracy:.2f}%")

st.subheader('Generate and Classify Images')
num_images = st.number_input('Enter number of images:', min_value=1, value=1000)

if st.button("Generate Images"):
    # Generate images
    st.text("Generating images...")
    generated_images = generate_images(num_images)
    
    # Classify images
    st.text("Classifying images...")
    generated_labels = classify_images(generated_images)
    
    # Save to HDF5 file
    st.text("Saving images and labels to HDF5 file...")
    save_to_hdf5(generated_images, generated_labels)

    st.text("Process completed.")
    real_images_data = []
    gen_images_data = []

    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Real Images')
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(0, 2):
            axes[0, i].imshow(real_images[i].squeeze(), cmap='bone')
            axes[0, i].axis('off')  # Hide axis
            real_images_data.append((real_images[i], real_labels[labels[i]][i]))
            axes[1, i].imshow(real_images[i+4].squeeze(), cmap='bone')
            axes[1, i].axis('off')  # Hide axis
            real_images_data.append((real_images[i+4], real_labels[labels[i]][i+4]))
        st.pyplot(fig)

    with col4:
        st.subheader('Generated Images')
        fig1, axes1 = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(0, 2):
            axes1[0, i].imshow(generated_images[i].squeeze(), cmap='bone')
            axes1[0, i].axis('off')  # Hide axis
            gen_images_data.append((generated_images[i], generated_labels[i]))
            axes1[1, i].imshow(generated_images[i+4].squeeze(), cmap='bone')
            axes1[1, i].axis('off')  # Hide axis
            gen_images_data.append((generated_images[i+4], generated_labels[i+4]))
        st.pyplot(fig1)

with st.sidebar: 
    st.sidebar.title('Generative X-Ray Synthesis')
    st.sidebar.info('This is the genration process of the generator during training')
    # Specify the directory containing the images
    image_dir = "generated_images"  
    image_files = os.listdir(image_dir)
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    image_placeholder = st.sidebar.empty()

    # Continuously update the displayed image with a delay
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_placeholder.image(image_path, use_column_width=True)
        time.sleep(0.75)

    st.write('for training and evaluation visualisation click here')
    if st.button('Start TensorBoard'):
        # Start TensorBoard session
        start_tensorboard("Tensorboard")  # Replace with your log directory
        time.sleep(5)
        webbrowser.open_new_tab("http://localhost:6006")  # Replace with your TensorBoard URL
        st.write("TensorBoard session started. Redirecting...")

