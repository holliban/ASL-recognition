# ASL-recognition

### About project
This project aims to achieve American Sign Language recognition. The main goal is also to achieve real-time webcam recognition.

### Dataset used
The dataset used can be found on Kaggle https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset/data.

### Example of data
<img width="2000" height="800" alt="dataset_preview" src="https://github.com/user-attachments/assets/3b4a723d-c395-4846-ac8e-66e28d35d165" />

### Specific Packages
In requirements.txt.

### Directory Structure
````
ASL-recognition/
│
├── data/ 
│   └── processed_combine_asl_dataset/
│           ├── 0/
│           ├── 1/
│           ├── ...
│           ├── a/
│           └── ...
│
│   └── dataset_split/
│           ├── test/
│           ├── trainval/
│
├── model/
│           ├── asl_cnn_1.keras
│           ├── asl_cnn_2.keras
│           ├── asl_cnn_2_2.keras
│           └── ...
│
├── png/ #if you want to save results, if not delete rows with savings of visualisations
├── train_model.ipynb
├── train_model_2.ipynb
├── real_time.py
└── requirements.txt
