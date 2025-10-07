# FONI-BM3000

## Handwritten Digit Recognizer

This project is a web application that recognizes handwritten digits. It uses a simple Artificial Neural Network (ANN) model built with Keras to predict the digit from an image drawn by the user on a canvas. The web interface is built with Flask.

### Features

* A simple and intuitive web interface to draw digits.
* Real-time prediction of the drawn digit.
* The backend is powered by a trained Artificial Neural Network.

### Technologies Used

* **Backend:**
    * Python
    * Flask
    * Keras (with TensorFlow backend)
    * NumPy
    * OpenCV
* **Frontend (assumed):**
    * HTML
    * CSS
    * JavaScript (for canvas drawing)
 
### Model
* **LGN (Lateral Geniculate Nucleus):** A relay center in the thalamus that receives visual information from the retina. In the model, initial layers might perform preprocessing and feature filtering, analogous to the LGN.
* **V1 (Primary Visual Cortex):** The first area of the cerebral cortex to receive and process basic visual information like edges, orientation, and spatial frequencies.
* **V2 (Secondary Visual Cortex):** Processes more complex visual attributes like shapes, contours, and object recognition.
The model is an Artificial Neural Network (ANN) built with Keras. The `Artificial_neural_network.ipynb` notebook provides details on the model architecture, training process, and performance. The model is saved as `ann_lgn_v1_v2_new.keras` and is loaded by the Flask application for making predictions.

The `app.py` script preprocesses the input image before feeding it to the model. The preprocessing steps include:
1.  Decoding the base64 image data from the request.
2.  Converting the image to grayscale.
3.  Resizing the image to 28x28 pixels.
4.  Inverting the colors.
5.  Normalizing the pixel values to be between 0 and 1.
6.  Flattening the image into a 1D array of 784 pixels.


---

## Word in 3D

This project is a Jupyter Notebook that visualizes word embeddings in 3D space. Word embeddings are a type of word representation that allows words with similar meanings to have a similar representation. This notebook uses a pre-trained Word2Vec model and visualizes the vector representations of words in a 3D plot.

### Features

* Loads a pre-trained Word2Vec model.
* Visualizes word vectors in 3D using PCA for dimensionality reduction.
* Interactive 3D plot to explore the relationships between words.

### Technologies Used

* Python
* Jupyter Notebook
* Gensim
* Scikit-learn (for PCA)
* Plotly (for 3D visualization)
