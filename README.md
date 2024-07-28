# Flower Classification with ConvNeXt Small

This project aims to classify different species of flowers using a Convolutional Neural Network (CNN) architecture called ConvNeXt Small. The dataset used for this project is the Oxford 17 Category Flower Dataset.

## Overview

This project demonstrates the use of the ConvNeXt Small architecture to classify images of flowers into 17 different categories. The key steps involved in the project are:

1. Data Loading and Preprocessing
2. Model Initialization
3. Model Training
4. Model Evaluation
5. Result Visualization

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- PyTorch
- timm (PyTorch Image Models)
- torchvision
- scikit-learn
- matplotlib

You can install the required packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Dataset

The dataset used is the Oxford 17 Category Flower Dataset, which consists of images of flowers categorized into 17 species.

## Installation

1. Clone the repository:

```
git clone https://github.com/abdulvahapmutlu/flower-classification.git
cd flower-classification
```

2. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Download the dataset and place it in the `data` directory.

## Usage

### Training the Model

To train the model, run:

```
python src/train.py
```

This script will:

- Load and preprocess the data
- Initialize the ConvNeXt Small model
- Train the model on the training set
- Validate the model on the validation set
- Save the trained model to the `models` directory

### Evaluating the Model

To evaluate the model on the test set, run:

```
python src/evaluate.py
```

This script will:

- Load the trained model
- Evaluate the model on the test set
- Print the classification report and confusion matrix

### Jupyter Notebook

You can also explore the project using the Jupyter notebook:

```
jupyter notebook notebooks/flower_classification.ipynb
```

The notebook includes all steps from data loading, preprocessing, model training, evaluation, and visualization of results.

## Results

The model achieved a test accuracy of 97.06%. You can see confusion matrix and plots in results section.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

 - Oxford 17 Flowers Dataset: The dataset used in this project is the Oxford 17 Flowers dataset, which is available on Kaggle.
 - PyTorch and torchvision: Libraries used for building and training the deep learning model.
 - Pretrained ConvNeXt Model: The ConvNeXt model, which was fine-tuned for this project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to improve this project.

## Contact

If you have any questions or feedback, please reach out to [abdulvahapmutlu1@gmail.com].
