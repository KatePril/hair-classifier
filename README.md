# Problem description
# Data
I used [Hair Type Dataset](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset) for training models for training and evaluation of the models of this project. This dataset can be downloaded from Kaggle via provided url. The dataset contains both male and female haircuts. Some images include full faces, while others focus specifically on the hair. Some photos are professionally-taken while others are home-taken.

The initial dataset contains:
- 514 images of curly hair
- 443 images of dreadlocks hair
- 217 images of kinky hair
- 488 images of straight hair
- 330 images of wavy hair

So as to split dataset into train/val/test directories run this command in the project directory:
```bash
python prepare_data.py
```
*Make sure you have replaced `input_directory` value with a correct path. After the data preprocessing, you will get the dataset split into three folders

The train directory contains:
- 436 images of curly hair
- 376 images of dreadlocks hair
- 184 images of kinky hair
- 414 images of straight hair
- 280 images of wavy hair

The val directory contains:
- 26 images of curly hair
- 22 images of dreadlocks hair
- 11 images of kinky hair
- 24 images of straight hair
- 16 images of wavy hair

The test directory contains:
- 52 images of curly hair
- 45 images of dreadlocks hair
- 22 images of kinky hair
- 50 images of straight hair
- 34 images of wavy hair

# Models

### model_1
The `model_1` architecture consists of the input layer, three convolutional layers, each followed by a max pooling layer, flatten layer and two dense layers.

### model_2
In the `model_2`'s architecture, the architecture of `model_1` was enhanced by a dropout layer, which was added between the dense layers.

### model_3
In the `model_3`'s architecture, the architecture of `model_2` was enhanced by a batch normalization layer, which was added after the first convolutional layer.

### model_4
The pre-trained Xception model was used as a base model for `model_4`. The Xception model a convolutional neural network with 71 layers, wich was trained on ImageNet dataset. A global average pooling layer and a dense layer were added to the architecture of the `model_4`.

### model_5
The pre-trained Xception model was also used as a base model for `model_5`. A global average pooling layer, two dense layers and a dropout layer between the dense layers were added to the architecture of the `model_5`.

### model_6
The pre-trained ResNet101 model was also used as a base model for `model_6`. The ResNet101 model a convolutional neural network with 101 layers, wich was trained on ImageNet dataset. A global average pooling layer and a dense layer were added to the architecture of the `model_6`.

### model_7
The pre-trained ResNet101 model was also used as a base model for `model_7`. A global average pooling layer, two dense layers and a dropout layer between the dense layers were added to the architecture of the `model_7`.

### Evaluation results
| **Model**  | **Accuracy** | **Loss** |
| -------- | -------- | ---- |
| model_1  | 0.244   | nan |
| model_2  |  0.244   | nan |
| model_3  |  0.323   | 1.398 |
| model_4  | 0.552   | 4.426  |
| model_5  | 0.672    | 0.756 |
| model_6  | 0.756   | 2.045  |
| model_7  | 0.687    | 0.864 |

### The final model
The `model_6`'s evaluation has shown the highest accuracy score so it was exported to the training script of the final model. The final model was trained on the original data only because additional training on the augmented data has proven to have a negative impact on the model's performance.

# Dependencies installation
All the necessary dependencies are listed in Pipfile. Install pipenv using the following command:
```bash
pip install pipenv
```
If you want to install all the dependencies, run:
```bash
pipenv install
```
If you want to install only the production dependencies, run:
```bash
pipenv install --ignore-pipfile --deploy
```
# Run project locally
First, clone the project repository
```bash
git clone https://github.com/KatePril/admission-prediction.git
```
```bash
cd hair-classifier
```
Run the `train.py` script:
```bash
python train.py
```
If you want to run Flask API deployment locally, type the following commands in the project main directory:
```bash
docker build -t <image-name> .
```
```bash
docker run -it --rm -p 9696:9696 <image-name>
```

If you want to run Gradio app locally, navigate to gradio directory:
```bash
cd gradio
```
Then, run the commands listed below:
```bash
docker build -t <image-name> .
```
```bash
docker run -it --rm -p 7860:7860 <image-name>
```

# Deployment
### Deployment demonstration
