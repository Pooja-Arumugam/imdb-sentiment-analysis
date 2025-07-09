# 🎬 IMDb Sentiment Analysis

#### Streamlit Link: https://imdb-sentiment-analysis-8fpgctwmtowjxwe7fwzcvf.streamlit.app/

This project implements a **Sentiment Analysis** model using a **Recurrent Neural Network (RNN)** to classify **IMDb movie reviews** as either **positive** or **negative**. The model is trained on the **IMDb dataset** available via **Keras** and leverages **deep learning** techniques for **natural language processing (NLP)**.

---

##  Problem Statement

The objective of this project is to build a **binary classification model** that can predict the **sentiment polarity** of a given movie review — **positive (1)** or **negative (0)**. This is a common task in **opinion mining** and **text classification**.

---

##  Technologies Used

- **Python 3.x**
- **TensorFlow** and **Keras**
- **NumPy**
- **Jupyter Notebook**
- **HDF5** for model storage
- **pip** for dependency management

---

##  Project Structure

```bash
imdb-sentiment-analysis/
│
├── .devcontainer/             # Configuration for VS Code Dev Containers
├── main.py                    # Main script for data processing and model training
├── mainn.py                   # Alternate version with similar functionality
├── prediction.ipynb           # Jupyter Notebook for sentiment predictions
├── requirements.txt           # Project dependencies
├── simple_rnn_imdb.h5         # Pretrained RNN model saved in HDF5 format
├── simplernn(dl).ipynb        # Notebook for model training and evaluation
└── README.md                  # Project documentation
```
---

##  Setup Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/Pooja-Arumugam/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```
### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Train the model
```bash
python main.py
```
### Step 4: Make predictions with pretrained model
```bash
jupyter notebook prediction.ipynb
```

## Dataset Description

- **Dataset**: IMDb Movie Reviews (via Keras)
- **Size**: 50,000 reviews (25k training, 25k testing)
- **Format**: Preprocessed as sequences of word indices
- **Labels**: `0` = Negative, `1` = Positive

---

## Model Architecture

This project uses the **Keras Sequential API** to build a simple RNN model:

- **Embedding Layer** – Transforms word indices into dense vectors
- **SimpleRNN Layer** – Learns sequential patterns from the text
- **Dense Layer with Sigmoid Activation** – Outputs sentiment probability

### Model Summary

```python
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32))
model.add(SimpleRNN(units=32))
model.add(Dense(1, activation='sigmoid'))
```

## Training Details

- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `accuracy`
- **Epochs**: 10
- **Batch Size**: 32

---

## Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Precision**, **Recall**, **F1-Score** _(optional in notebook)_

---

## Notebooks

- `simplernn(dl).ipynb` – End-to-end training and evaluation
- `prediction.ipynb` – Load model and predict sentiment for custom text

---

## Example Prediction

```python
# Load and use the model
model = load_model('simple_rnn_imdb.h5')
input_text = "The movie was fantastic!"
```
# Preprocess and predict...
``` python
 Predicted Sentiment: Positive (Probability: 0.93)
```



