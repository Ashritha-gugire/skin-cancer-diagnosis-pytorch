# skin-cancer-diagnosis-pytorch
## Image processing algorithm using Convolutional Neural Networks (CNNs) 

This project focuses on developing a deep learning model for the classification of skin lesions, with the ultimate objective of diagnosing various skin conditions. 
Key aspects of the project include:

1.Utilizing a dataset that includes patient/demographic details such as gender, age, lesion location on the body, along with images and lesion IDs.
2.Data preprocessing and exploration, including filtering image files based on file extension and mapping lesion types to their corresponding categories for analysis and model training.
3.Implementing a pretrained ResNetX101 model with data augmentation techniques to improve robustness and avoid overfitting.
4.Dividing the dataset into validation and training sets to assess model performance.
5.Tracking metrics like training loss and validation accuracy during the training phase.
6.Achieving improved accuracy and decreasing loss over epochs.


![image](https://github.com/user-attachments/assets/0ce295ff-dddc-47ae-93bd-09e10e17a167)
  



### Dataset
The HAM10000 dataset comprises approximately 10,000 labeled images depicting skin lesions. It includes:

Patient/demographic details (gender, age)
Lesion location on the body
Images and Lesion IDs

Dataset Source: HAM10000 on Kaggle https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data 

**Original Challenge**
This project is inspired by the ISIC 2018 challenge:

"ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection"
Hosted by the International Skin Imaging Collaboration (ISIC)
Original Challenge: https://challenge2018.isic-archive.com A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368 https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 
__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Introduction
This project aims to develop an image processing algorithm using Convolutional Neural Networks (CNNs) to identify specific types of skin cancer from clinical images. Utilizing a ResNetX101 architecture, the model classifies skin lesions into seven categories: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Vascular lesions, and Dermatofibroma.

### Research Question
This project is part of a broader research initiative addressing the question:
"How can the applications of Hidden Markov Models (HMMs) in the analysis of genomic data enhance our capacity to detect dynamic gene expression patterns linked to cancer progression, and what valuable insights can be gleaned from these HMM-based analyses regarding the genetic mechanisms driving cancer development?"

While our current project focuses on image-based diagnosis using CNNs, future work may incorporate HMM-based analysis of genomic data to provide a more comprehensive understanding of skin cancer progression and development.

### Motivation
Skin cancer is one of the most common types of cancer, with a significant impact on global health. Early and accurate diagnosis is crucial for effective treatment. This project seeks to aid medical professionals in diagnosing skin cancer by leveraging deep learning techniques, potentially reducing diagnostic errors and improving patient outcomes.
The increasing prevalence of genetically inherited diseases necessitates sophisticated genetic data analysis. By combining image-based diagnosis with genomic data analysis, we aim to improve our understanding of the genetic basis of skin cancer and other genetically transmitted diseases.

**_Methodology_**


![image](https://github.com/user-attachments/assets/5047e6b3-9989-4697-9ef8-615972dbfa9d)

Data Preprocessing:

Image filtering based on file extension
Mapping lesion types to categories
Image resizing and normalization
Handling class imbalance through data augmentation


Model Architecture:

ResNetX101 (pretrained)
Custom fully connected layers for classification


Training:

PyTorch framework
Data augmentation techniques: random flipping, rotation, color jittering
Dataset division into training and validation sets


Evaluation:

Tracking training loss and validation accuracy
Accuracy, precision, recall, F1-score
Confusion matrix analysis


**Results**

Improved accuracy and decreasing loss over epochs (specific metrics to be added)
High performance on certain classes (details to be provided)
Areas for improvement identified in specific classes

Dependencies

Python 3.7+
PyTorch
torchvision
OpenCV (cv2)
PIL
tqdm

**Future Work**

Address class imbalance further
Experiment with other CNN architectures
Incorporate additional data sources
Develop a user-friendly interface for clinical use
Integrate HMM-based analysis of genomic data to complement image-based diagnosis
Explore the application of HMMs in detecting dynamic gene expression patterns linked to cancer progression
Investigate the genetic mechanisms driving skin cancer development using insights from HMM-based analyses


License
This project is licensed under the MIT License - see the LICENSE.md file for details.

_Acknowledgments_
HAM10000 dataset providers
PyTorch community
George Mason University
ISIC for the original challenge and dataset

_**References**_

Zhang, K., Yang, Y., Devanarayan, V., Xie, L., Deng, Y., & Donald, S. (n.d.). A Hidden Markov model-based algorithm for identifying tumor subtype. Retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3287492/ 

Wojtowicz, D., Sason, I., Huang, X., Kim, Y. A., & Leiserson, M. D. M. (2019). Hidden Markov models lead to higher resolution maps of mutation signature activity in cancer. Genome Medicine, 11(1). https://doi.org/10.1186/s13073-019-0659-1 

Elandt-Johnson, R. C. (1971). Probability models and statistical methods in genetics. New York: Wiley. 

Yu, A. (2018, December 16). Building a Skin Lesion Classification Web App Using Keras and TensorFlow.js to classify seven types of skin lesions. Towards Data Science. GitHub Repository 

Tajerian, A. (n.d.). A new machine-learning based diagnostic tool for differentiation of dermatoscopic skin cancer images. GitHub Repository
