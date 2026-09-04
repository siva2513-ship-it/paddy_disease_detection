#  Paddy Disease Detection

An end-to-end machine learning project for identifying diseases and pests affecting paddy crops from leaf images.

The main goal of this project is to explore how deep learning can be combined with traditional machine learning techniques to build an image classification pipeline for paddy disease detection. The project starts with a fine-tuned **ResNet34 CNN** and then uses the learned CNN representations as features for classical machine learning models.

The current implementation covers **10 paddy disease/pest categories**, with experiments conducted using ResNet34, SVM, KNN, Random Forest, and XGBoost.


##  Project Objective

Paddy crops can be affected by several diseases and pests that produce visible symptoms on leaves and other parts of the plant. Early identification can help farmers take appropriate action before the problem spreads.

In this project, image classification is used to recognize different paddy diseases and pests from photographs.

The project follows two main ideas:

1. Train a CNN directly on paddy images and evaluate its classification performance.
2. Reuse the CNN's learned visual representation as a feature extractor and compare several traditional machine learning classifiers.

This allows us to compare **end-to-end deep learning** against **classical ML applied to deep-learning features**.


##  Overall ML Pipeline


                    Paddy Leaf Images
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                    ResNet34 CNN
                  Transfer Learning
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Direct Classification      Feature Extraction
                                      │
                                      ▼
                              512-D Feature Vector
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
                   SVM               KNN          Random Forest
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
                                  XGBoost

The ResNet34 model is first fine-tuned on the original images. Its learned representation is then reused to generate a **512-dimensional feature vector** for each image.

These feature vectors are used as input to SVM, KNN, Random Forest, and XGBoost.


#  Dataset

The project uses the **Paddy Doctor** dataset available through Kaggle.

The dataset contains:

* **10,407 labelled training images**
* **3,469 unlabelled test images**
* **10 classes**
* All training images are used for the project's train/validation experiments.

### Dataset Classes

| Class                    | Training Images |
| ------------------------ | --------------: |
| Normal                   |           1,764 |
| Blast                    |           1,738 |
| Hispa                    |           1,594 |
| Dead Heart               |           1,442 |
| Tungro                   |           1,088 |
| Brown Spot               |             965 |
| Downy Mildew             |             620 |
| Bacterial Leaf Blight    |             479 |
| Bacterial Leaf Streak    |             380 |
| Bacterial Panicle Blight |             337 |

The dataset is somewhat imbalanced, with some classes having considerably more images than others.


# Train / Validation Split

All experiments use the same random split so that the models can be compared fairly.

Total labelled images : 10,407
Training images       : 8,326
Validation images     : 2,081
Split                 : 80 / 20
Random seed            : 42

The same split is recreated in the feature extraction and classical ML notebooks.

This means that the reported model results are directly comparable because they are evaluated on the same validation images.


#  Models

The project currently contains five models.

| Model   | Type          | Input                   |
| ------- | ------------- | ----------------------- |
| Model 1 | ResNet34 CNN  | Raw images              |
| Model 2 | SVM           | 512-D ResNet34 features |
| Model 3 | KNN           | 512-D ResNet34 features |
| Model 4 | Random Forest | 512-D ResNet34 features |
| Model 5 | XGBoost       | 512-D ResNet34 features |

An important point is that **Models 2–5 are not trained directly on images**.

They all use the feature representation learned by Model 1.


#  ResNet34 — CNN Baseline

ResNet34 is the foundation of the project.

A pretrained ResNet34 network is fine-tuned on the paddy disease images using the FastAI/PyTorch framework.

### Training configuration

* Architecture: ResNet34
* Framework: FastAI / PyTorch
* Pretrained weights: ImageNet
* Image resize: 480px
* Final training size: 224px
* Augmentation: `aug_transforms`
* Minimum scale: `0.75`
* Epochs: 50
* Learning rate: `0.005`
* Mixed precision: FP16
* Train/validation split: 80/20
* Random seed: 42

### Verified Result

**Validation Accuracy: 98.03%**

This is currently the strongest result in the project.

The ResNet34 model is also used as the feature extractor for the classical ML experiments.


##  Support Vector Machine

The SVM receives the 512-dimensional feature vectors generated by ResNet34.

Configuration:

kernel = RBF
C = 10
gamma = scale
probability = True
random_state = 42

### Verified Result

**Validation Accuracy: 76.93%**

SVM achieved the highest accuracy among the four classical ML models tested.

More details:

[`models/model_2_svm/README.md`](models/model_2_svm/README.md)


##  K-Nearest Neighbors

KNN also operates on the 512-dimensional ResNet34 features.

Configuration:

n_neighbors = 5
weights = distance
metric = euclidean
n_jobs = -1

### Verified Result

**Validation Accuracy: 70.30%**

KNN was extremely fast to fit because it does not perform a conventional model-training process, but prediction is slower because it compares validation samples against the training feature set.

More details:

[`models/model_3_knn/README.md`](models/model_3_knn/README.md)


#  Random Forest

Random Forest was trained using the same 512-dimensional feature representation.

Configuration:

n_estimators = 300
class_weight = balanced
random_state = 42
n_jobs = -1

### Verified Result

**Validation Accuracy: 69.73%**

Random Forest produced the lowest accuracy among the four classical models in this experiment, although its inference time was relatively low.


#   XGBoost

XGBoost was the final classical ML model in the current comparison.

Configuration:

n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
random_state = 42

### Verified Result

**Validation Accuracy: 76.17%**

XGBoost performed very close to SVM in terms of accuracy, but required significantly more training time in this experiment.


#  Feature Extraction

After training ResNet34, the network is reused as a feature extractor.

Instead of using the final 10-class prediction layer, the project takes the learned representation immediately before the final classifier.

This produces:

1 image
   ↓
ResNet34
   ↓
512 numbers
   ↓
512-dimensional feature vector


The extracted features are stored as NumPy arrays:

X_train.npy
y_train.npy
X_valid.npy
y_valid.npy

with:

X_train : 8326 × 512
X_valid : 2081 × 512

These feature files are then used by the SVM, KNN, Random Forest, and XGBoost notebooks.


#  Model Comparison

The following results were obtained on the same **2,081-image validation set**.

| Model         |   Accuracy | Macro F1 | Weighted F1 | Training Time | Inference Time |
| ------------- | ---------: | -------: | ----------: | ------------: | -------------: |
| **ResNet34**  | **98.03%** |      N/A |         N/A |           N/A |            N/A |
| **SVM**       | **76.93%** |   0.7281 |      0.7657 |      102.67 s |        10.23 s |
| **XGBoost**   | **76.17%** |   0.7277 |      0.7597 |      403.45 s |         0.17 s |
| KNN           |     70.30% |   0.6599 |      0.7007 |        0.01 s |         3.30 s |
| Random Forest |     69.73% |   0.6478 |      0.6908 |       47.01 s |         0.19 s |

### What the results tell us

The fine-tuned ResNet34 clearly performed better than the classical models.

The best classical model was SVM with **76.93% accuracy**, followed closely by XGBoost at **76.17%**.

This experiment suggests that, for this dataset and configuration, allowing the CNN to learn the classification task end-to-end was considerably more effective than freezing its representation and using a separate classical classifier.

The classical models are still useful because they provide a meaningful comparison and show how different algorithms behave when working with learned CNN features.


# Evaluation Note

The reported accuracy values are **validation results**.

The Kaggle test set contains 3,469 images but does not provide class labels in the downloaded test-image structure used by the notebooks. Therefore, it cannot be used to calculate a meaningful classification accuracy locally.

An attempted evaluation of that test set exists in the ResNet34 notebook, but the resulting score is not valid.

Therefore:

> **No model in the current repository has a verified accuracy on the official Kaggle test set.**

The **98.03% ResNet34 result** and the classical-model results are based on the project's held-out validation set.


#  Notebook Workflow

The notebooks are designed to be run in stages.

resenet34_baseline_.ipynb
          │
          ▼
Feature_Extraction.ipynb
          │
          ├──────────────┬──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
        SVM             KNN       Random Forest     XGBoost

### Phase 1

`resenet34_baseline_.ipynb`

Trains and exports the ResNet34 CNN.

### Phase 2

`Feature_Extraction.ipynb`

Loads the trained ResNet34 model and generates the reusable feature vectors.

### Phase 3

The following notebooks can then be run independently:

* `SVM_model.ipynb`
* `KNN_model.ipynb`
* `RandomForest_model.ipynb`
* `XGBoost_model.ipynb`


#  Repository Structure

paddy_disease_detection/
│
├── README.md
├── .gitattributes
│
├── colab_notebooks/
│   ├── README.md
│   ├── resenet34_baseline_.ipynb
│   ├── Feature_Extraction.ipynb
│   ├── SVM_model.ipynb
│   ├── KNN_model.ipynb
│   ├── RandomForest_model.ipynb
│   └── XGBoost_model.ipynb
│
└── models/
    │
    ├── model_1_resenet34/
    │   ├── README.md
    │   ├── resnet34_paddy_baseline.pkl
    │   └── resnet34_paddy_baseline.pth
    │
    ├── model_2_svm/
    │   ├── README.md
    │   └── model_2_svm.pkl
    │
    ├── model_3_knn/
    │   ├── README.md
    │   └── model_3_knn.pkl
    │
    ├── model_4_random_forest/
    │   ├── README.md
    │   └── model_4_random_forest.pkl
    │
    └── model_5_xgboost/
        ├── README.md
        └── model_5_xgboost.pkl


# Model Storage

The trained models include several large binary files.

Git LFS is used to store these large model files.

After cloning the repository, install and initialize Git LFS:

git lfs install
git lfs pull

The model files include:

* ResNet34 `.pkl`
* ResNet34 `.pth`
* SVM `.pkl`
* KNN `.pkl`
* Random Forest `.pkl`
* XGBoost `.pkl`

The `.pth` and Random Forest files are particularly large and therefore require Git LFS.

---

#  Security

The notebooks use Kaggle authentication to download the dataset.

**Never commit your Kaggle API credentials to GitHub.**

In particular, files such as:

kaggle.json
API keys
access tokens
passwords
credentials

should never be uploaded to the repository.


#  Current Status

###  Completed

* Paddy disease dataset preparation
* Dataset inspection and validation
* ResNet34 transfer-learning baseline
* ResNet34 feature extraction
* 512-dimensional feature generation
* SVM classification
* KNN classification
* Random Forest classification
* XGBoost classification
* Model evaluation on a common validation split
* Model comparison
* Model storage using Git LFS


#  Important Project Insight

One of the main observations from the current experiments is the difference between **feature extraction + classical ML** and **end-to-end CNN fine-tuning**.

The ResNet34 model achieved **98.03% validation accuracy**, while the best classical model, SVM, achieved **76.93%** using the ResNet34 feature representation.

This does not mean classical ML is ineffective. Instead, it shows that for this particular dataset and the configurations tested here, allowing the CNN to continue learning specifically for the classification task produced a much stronger result.

The classical models therefore serve an important role in this project: they provide a baseline for understanding how much classification performance can be obtained from the CNN's learned features without fine-tuning a separate classifier end-to-end.

