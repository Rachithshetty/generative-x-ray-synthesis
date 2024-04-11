# Generative X-ray Synthesis

This project aims to generate synthetic X-ray images using generative adversarial networks (GANs) and then use a classification model to assign labels to the generated images. The generated images are saved along with their corresponding labels to an HDF5 file for further analysis.

## Introduction

Generative X-ray Synthesis is a project that leverages machine learning techniques to generate synthetic X-ray images and classify them using a pre-trained model. It provides a platform for researchers and medical professionals to explore the potential applications of generative models in medical imaging.

## Installation

To install and run the project locally, follow these steps:

1. Clone the repository: [git clone](https://github.com/Rachithshetty/generative-x-ray-synthesis.git)
2. Navigate to the project directory: `cd generative-x-ray-synthesis`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

Once the project is installed, you can use it to generate synthetic X-ray images and classify them using the provided models. Run the Streamlit app by executing the following command: `streamlit run app.py`

This will launch a local web application where you can interact with the models and visualize the results.

## Docker

Alternatively, you can use Docker to run the project in a containerized environment. Build the Docker image using the provided Dockerfile: `[docker build](https://github.com/Rachithshetty/generative-x-ray-synthesis/blob/main/Dockerfile) -t generative-x-ray-synthesis .`

Then, run the Docker container: `[docker run](https://github.com/Rachithshetty/generative-x-ray-synthesis/blob/main/Dockerfile) -it --gpus all --rm generative-x-ray-synthesis`
(make sure to modify the Dockerfile according to your requirement)

This will start the Streamlit app inside the Docker container, allowing you to access it via your web browser.

## Dataset

The NIH ChestXray14 dataset is a collection of chest X-ray images used for various medical image analysis tasks, particularly for detecting thoracic diseases. The dataset consists of 112,120 frontal-view X-ray images from 30,805 unique patients, annotated with up to 14 different thoracic pathology labels.

### Download

You can download the NIH ChestXray14 dataset from the following link:

[NIH ChestXray14 Dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

Please note that you may need to sign in to Kaggle and agree to their terms and conditions to access the dataset.

## Contributing

If you'd like to contribute to the project, feel free to submit bug reports, feature requests, or pull requests through GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
