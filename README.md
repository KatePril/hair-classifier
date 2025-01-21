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
