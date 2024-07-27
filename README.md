# LyubomirT's Toxicity Detector Model v17

Welcome to LyubomirT's Toxicity Detector repository! This project contains a Jupyter notebook that demonstrates how to train and use a lightweight BERT-based model for detecting toxicity in text. 

## Overview

This notebook utilizes the BERT model to classify comments into different toxicity categories based on the "Severity of Toxic Comments" dataset. The model is implemented using the `transformers` library and `torch`. 

### Features

- **Model Training**: Instructions on how to train the BERT model on the toxicity dataset.
- **Inference**: Methods to use the pre-trained model for making predictions on new text.
- **Pre-trained Weights**: Option to download pre-trained model weights for quick inference.

## Getting Started

### Prerequisites

Before running the notebook, make sure you have the following Python packages installed:

- `pandas`
- `transformers`
- `torch`
- `tqdm`
- `scikit-learn`

You can install these packages using pip:

```bash
pip install pandas transformers torch tqdm scikit-learn
```

### Training the Model

1. **Run the Notebook**: Start by running all the cells in the notebook from the beginning. 
2. **GPU Requirement**: Training the model efficiently requires a GPU. Ensure you have access to one for best performance.

### Quick Inference

If you prefer to skip the training process and test the model directly:

1. **Download Pre-trained Weights**: Obtain the trained model weights from the [release page](https://github.com/LyubomirT/bert-toxicity-detection-model/releases).
2. **Upload Weights**: Upload the model weights to the notebook.
3. **Run Inference**: Navigate to the "Inference" section in the notebook. You can input text and observe the model's predictions.

### Dataset

The model is trained on the "Severity of Toxic Comments" dataset, which includes labels such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The dataset consists of comments from Wikipedia's talk page edits.

### Model Details

- **Model Architecture**: Utilizes BERT (`bert-base-uncased`) with a classification head for toxicity detection.
- **Training Configuration**: The model is trained using the AdamW optimizer with a learning rate of 2e-5 and employs mixed precision training and gradient accumulation for efficiency.

### Inference

To perform inference:

1. **Model Loading**: The notebook includes code to load the pre-trained model weights.
2. **Prediction Function**: Use the `predict_toxicity` function to evaluate the toxicity of input text.
3. **Output**: The function returns a dictionary with the toxicity probabilities for each category.

---

Basically, this means that you can download the weights and run all cells + the last one to test the model on your own text. Also, you can train the model on your own dataset by following the instructions in the notebook.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and improvements are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.