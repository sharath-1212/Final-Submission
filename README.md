
# ♻️ E-Waste Image Classification Using EfficientNetV2B3

A deep learning-based system for automated classification of electronic waste images using transfer learning with EfficientNetV2B3, deployed via Streamlit for real-time interaction.

---

## 📌 Project Overview

Electronic waste (e-waste) poses a significant threat to the environment and public health due to improper disposal and recycling. This project aims to automate the classification of e-waste images using a convolutional neural network (CNN) architecture—**EfficientNetV2B3**—to support sustainable and scalable waste management.

---

## 🎯 Objectives

- Automatically classify e-waste images into 10 categories.
- Improve classification accuracy using transfer learning.
- Deploy a real-time, user-friendly web app with feedback logging.

---

## 🧠 Model Summary

- **Architecture**: EfficientNetV2B3 (pretrained on ImageNet)
- **Approach**: Transfer learning + fine-tuning
- **Input Size**: 300x300 RGB images
- **Output**: 10 waste classes
- **Test Accuracy**: 96%

---

## 🗂️ Dataset

- **Source**: [Kaggle - E-Waste Image Dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)
- **Classes**: Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player, Printer, Television, Washing Machine
- **Structure**: Train / Validation / Test (Balanced - 240/30/30 per class)

---

## 🛠️ Technologies Used

| Tool/Library       | Purpose                                         |
|--------------------|-------------------------------------------------|
| Python 3.x         | Core programming language                       |
| TensorFlow & Keras | Model building, training, and evaluation        |
| EfficientNetV2B3   | Transfer learning backbone                      |
| Jupyter Notebook   | Interactive development & training              |
| Pillow (PIL)       | Image loading and preprocessing                 |
| Streamlit          | Real-time web interface                         |
| Gradio (optional)  | Quick local testing interface                   |
| NumPy              | Array and image data processing                 |
| CSV Logging        | Records flagged images with timestamp & output  |

---

## 🚀 Streamlit Web App Features

- Upload e-waste image for prediction.
- See predicted class with confidence score.
- View progress bar and class description.
- Flag uncertain/misclassified images for improvement.
- All flagged entries are logged in a `dataset.csv`.

To run the app locally:

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Training Summary

- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: SparseCategoricalCrossentropy
- **Epochs**: 15
- **EarlyStopping**: Patience = 3, Restore best weights
- **Accuracy Achieved**: 96% on test set

---

## 📈 Evaluation Metrics

- Accuracy: 96%
- Test Loss: 0.1073
- F1-Scores: 0.92 – 1.00 across all classes
- Precision/Recall: High across all categories
- Confusion matrix: Minimal misclassification

---

## 🏁 Project Flow

1. Dataset Collection and Preprocessing  
2. Model Design with EfficientNetV2B3  
3. Training and Fine-Tuning  
4. Evaluation with Test Data  
5. Streamlit Deployment  
6. User Feedback Logging

---



## 🌍 Real-World Impact

- Enables automated, scalable e-waste sorting
- Reduces manual labor and misclassification
- Supports environmental sustainability goals
- Educates users and promotes responsible recycling

---

## 📌 Future Enhancements

- Add multi-object detection for mixed e-waste
- Deploy on cloud or edge devices (e.g., Raspberry Pi)
- Improve model with more labeled data and active learning
- Build a mobile app version of the interface

---

## 🙏 Acknowledgements

- [Kaggle Dataset by Akshat](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)
- Streamlit, TensorFlow, and EfficientNet teams
- Freepik.com for icons

---

## 👤 Author

**Sharath Adepu**  
*AI & Sustainability Enthusiast*
