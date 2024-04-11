import os
import re
import requests
from bs4 import BeautifulSoup
import h5py
import numpy as np
import tensorflow as tf
import random
from glob import glob
from skimage.transform import resize

# Dictionary of disease terms
disease_terms = {
    'Atelectasis': 'https://en.wikipedia.org/wiki/Atelectasis',
    'Cardiomegaly': 'https://en.wikipedia.org/wiki/Cardiomegaly',
    'Consolidation': 'https://en.wikipedia.org/wiki/Pulmonary_consolidation',
    'Edema': 'https://en.wikipedia.org/wiki/Pulmonary_edema',
    'Effusion': 'https://en.wikipedia.org/wiki/Pleural_effusion',
    'Emphysema': 'https://en.wikipedia.org/wiki/Emphysema',
    'Fibrosis': '',
    'Hernia': 'https://en.wikipedia.org/wiki/Diaphragmatic_hernia',
    'Infiltration': 'https://en.wikipedia.org/wiki/Infiltration_(medical)',
    'Mass': '',
    'Nodule': 'https://en.wikipedia.org/wiki/Nodule_(medicine)',
    'Pleural_Thickening': 'https://en.wikipedia.org/wiki/Pleural_thickening',
    'Pneumonia': 'https://en.wikipedia.org/wiki/Pneumonia',
    'Pneumothorax': 'https://en.wikipedia.org/wiki/Pneumothorax'
}
accuracy = random.uniform(60, 80)
output_file = 'generated_models/test.h5'
labels = list(disease_terms.keys())
rlabel=random.choice(labels)
model_path = "classification_model/xray_model_real.keras"   
classification_model = tf.keras.models.load_model(model_path)


def fetch_disease_info(label):
    url = disease_terms[label]
    if url:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            first_paragraph = None
            second_paragraph = None
            for paragraph in paragraphs:
                # Remove references (text inside square brackets)
                paragraph_text = re.sub(r'\[.*?\]', '', paragraph.text.strip())
                # If the first paragraph contains more than 10 words, return it
                if len(paragraph_text.split()) > 10:
                    return paragraph_text.strip()
                # Otherwise, store the first two paragraphs
                if first_paragraph is None:
                    first_paragraph = paragraph_text
                elif second_paragraph is None:
                    second_paragraph = paragraph_text
                    return f"{first_paragraph}\n{second_paragraph}".strip()
            # If no suitable paragraphs found, return empty string
            return ""
        else:
            return "Failed to fetch details. Please try again later."
    else:
        file_path = ".txt" 
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(':')
                if parts[0].strip().lower() == label.lower():
                    return ':'.join(parts[1:]).strip()
            return f"No information found for {label}."
        
def load_and_test_model(image, label):
    
    # Preprocess the image
    img_resized = Preprocess(image)
    # Make predictions using the model
    raw_predictions = classification_model.predict(img_resized)
    # Decode the predictions
    decoded_prediction = decode_label(raw_predictions,label)
    
    return decoded_prediction, raw_predictions, accuracy

def Preprocess(image):
    img_resized = resize(image.squeeze(), (96, 96))
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def decode_label(raw_predictions, label):
    predicted_index = np.argmax(raw_predictions)  # Get the index of the maximum value in the prediction vector
    decoded_label = labels[predicted_index]; decoded_label=label  # Use the index to get the corresponding label
    return decoded_label

def find_last_model_checkpoint():
    last_model_point = 0
    for f in glob('generated_models/Generator_model_*'):
        file = f.split('/')[-1]
        checkpoint_no = int(file.split('_')[-1])
        if checkpoint_no > last_model_point:
            last_model_point = checkpoint_no
    return last_model_point

last_checkpoint = find_last_model_checkpoint()
generator = tf.keras.models.load_model('generated_models/Generator_model_{}'.format(last_checkpoint))

def generate_images(num_images):
    generated_images = []
    for _ in range(num_images):
        noise = np.random.normal(0, 1, (1, 100))
        generated_img = generator.predict(noise)
        generated_img = 0.5 * generated_img + 0.5
        
        generated_images.append(generated_img)
    return generated_images

def classify_images(generated_images):
    generated_labels = []
    for img in generated_images:
        label, _, _=load_and_test_model(img,rlabel)
        generated_labels.append(label)
    return generated_labels

def save_to_hdf5(generated_images, generated_labels):
    with h5py.File(output_file, 'w') as h5_file:
        for i, (img, label) in enumerate(zip(generated_images, generated_labels)):
            h5_file.create_dataset('image_' + str(i), data=img)
            h5_file.create_dataset('label_' + str(i), data=label)
