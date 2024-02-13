import os
import re
import PyPDF2
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from sklearn.model_selection import train_test_split

# Preprocessing Text Data

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    str: Extracted text from the PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

def clean_text(text):
    """
    Clean the extracted text by removing special characters, unwanted symbols, and noise.
    
    Args:
    text (str): The text to be cleaned.
    
    Returns:
    str: Cleaned text.
    """
    cleaned_text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Remove extra spaces
    cleaned_text = cleaned_text.strip()  # Remove leading and trailing spaces
    return cleaned_text

def preprocess_text(text):
    """
    Preprocess the text by converting it to lowercase and removing extra spaces.
    
    Args:
    text (str): The text to be preprocessed.
    
    Returns:
    str: Preprocessed text.
    """
    preprocessed_text = text.lower()  # Convert to lowercase
    preprocessed_text = re.sub(r"\s+", " ", preprocessed_text)  # Remove extra spaces
    return preprocessed_text

# Training the Model

def load_pretrained_model(model_name):
    """
    Load a pre-trained transformer model for text generation.
    
    Args:
    model_name (str): Name of the pre-trained model.
    
    Returns:
    transformer model: Pre-trained transformer model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    return tokenizer, model

def fine_tune_model(model, tokenizer, X_train, y_train, X_val, y_val, epochs=3, batch_size=8):
    """
    Fine-tune the pre-trained model on the novel dataset.
    
    Args:
    model (transformer model): Pre-trained transformer model.
    tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): Tokenizer for the model.
    X_train (list): List of training input sequences.
    y_train (list): List of corresponding target sequences for training.
    X_val (list): List of validation input sequences.
    y_val (list): List of corresponding target sequences for validation.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.
    """
    train_dataset = create_dataset(X_train, y_train, tokenizer)
    val_dataset = create_dataset(X_val, y_val, tokenizer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(train_dataset.shuffle(1000).batch(batch_size),
              epochs=epochs,
              validation_data=val_dataset.batch(batch_size))

def create_dataset(X, y, tokenizer):
    """
    Create a TensorFlow dataset from input and target sequences.
    
    Args:
    X (list): List of input sequences.
    y (list): List of target sequences.
    tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): Tokenizer for the model.
    
    Returns:
    tf.data.Dataset: TensorFlow dataset.
    """
    input_ids = tokenizer(X, padding=True, truncation=True, return_tensors="tf")["input_ids"]
    labels = tokenizer(y, padding=True, truncation=True, return_tensors="tf")["input_ids"]
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, labels))
    return dataset

# Example usage:

# Specify paths to PDF files containing novels
novel_paths = ["novels_folder/" + file for file in os.listdir("novels_folder")]

# Extract text from PDF files
novel_texts = [extract_text_from_pdf(path) for path in novel_paths]

# Clean and preprocess text data
cleaned_texts = [clean_text(text) for text in novel_texts]
preprocessed_texts = [preprocess_text(text) for text in cleaned_texts]

# Split the preprocessed data into training and validation sets
X_train, X_val = train_test_split(preprocessed_texts, test_size=0.1, random_state=42)

# Load pre-trained GPT-3.5 model
tokenizer, model = load_pretrained_model("gpt2-medium")

# Fine-tune the model on the novel dataset
fine_tune_model(model, tokenizer, X_train, X_train, X_val, X_val)

