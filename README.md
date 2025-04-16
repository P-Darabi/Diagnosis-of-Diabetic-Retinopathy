# Diagnosis of Diabetic Retinopathy

This project focuses on diagnosing Diabetic Retinopathy (DR) from retinal fundus images using Convolutional Neural Networks (CNN) in PyTorch. The goal of this model is to assist in the early diagnosis of Diabetic Retinopathy, potentially preventing serious vision issues in diabetic patients.

## Project Contents

- **`train.py`**: Script for training the CNN model to detect Diabetic Retinopathy.
- **`evaluate.py`**: Script for evaluating the model and testing its predictions' accuracy.
- **`model.py`**: Defines the CNN architecture used to process and classify retinal fundus images.
- **`utils.py`**: Helper functions for loading and preprocessing data.
- **`requirements.txt`**: List of dependencies required to run the project.

## Dataset

The model is built using the [Diagnosis of Diabetic Retinopathy](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy) dataset, which contains retinal fundus images with labels indicating the severity of Diabetic Retinopathy.

### Useful Links:

- **Dataset**: [Diagnosis of Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy)
- **Code**: [Diagnosis of Diabetic Retinopathy by CNN (PyTorch)](https://www.kaggle.com/code/pkdarabi/diagnosis-of-diabetic-retinopathy-by-cnn-pytorch)

## Model Architecture

This model uses a Convolutional Neural Network (CNN) with multiple convolutional layers followed by fully connected layers. The network automatically extracts features from retinal fundus images and classifies them into different categories of Diabetic Retinopathy severity.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/P-Darabi/Diagnosis-of-Diabetic-Retinopathy.git
    cd Diagnosis-of-Diabetic-Retinopathy
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the following command:

```bash
python train.py --data_path /path/to/dataset --epochs 25
```

### Evaluating the Model

To evaluate the model, run:

```bash
python evaluate.py --model_path /path/to/trained_model.pth --test_data /path/to/test_data
```

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request with your proposed modifications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
