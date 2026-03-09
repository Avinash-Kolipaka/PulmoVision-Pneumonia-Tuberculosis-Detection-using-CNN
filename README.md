# PulmoVision : CNN-for-Pneumonia-Tuberculosis-Detection

## 📌 Introduction  
**PulmoVision** is a deep learning–based computer vision project designed to detect **Tuberculosis** and **Pneumonia** from **chest X-ray images**.  
Using a **Convolutional Neural Network (CNN)**, the model classifies X-rays into three categories: **Normal, Pneumonia, and Tuberculosis**.  
This project supports medical imaging research by providing an automated system for faster and more accurate preliminary diagnosis. 
---

## 📂 Dataset Choice  
The dataset used is the **[Imbalanced Tuberculosis and Pneumonia Dataset](https://www.kaggle.com/datasets/roshanmaur/imbalanced-tuberculosis-and-pnuemonia-dataset)** available on Kaggle.  

- **Total images:** 15,121  
- **Classes:**  
  - Normal → 9,188 images  
  - Pneumonia → 4,145 images  
  - Tuberculosis → 1,788 images  
- **Image size range:** 1,000×600 up to 4,800×4,800  
- **Dataset split:** 70% Train | 15% Validation | 15% Test  

---

## 🔄 Workflow  

1. **Data Acquisition**  
   - Downloaded via `kagglehub` or Kaggle input path.  
   - Removed unused class labels (`universal_test`).  

2. **Preprocessing & Augmentation**  
   - Resized all images to **224×224**.  
   - Training augmentations: random rotations, flips, brightness/contrast jitter.  
   - Normalization applied to all images.  

3. **Model Architecture (CNN)**  
   - 3 Convolutional + MaxPooling layers (32 → 64 → 128 filters).  
   - Fully connected layers:  
     - Flatten → Dense(256) → Dropout(0.5) → Dense(3).  
   - Loss: **CrossEntropyLoss**  
   - Optimizer: **Adam (lr=1e-4)**  

4. **Training**  
   - Trained for 5 epochs, batch size = 32.  
   - Monitored accuracy & loss for training and validation.  

5. **Evaluation**  
   - Evaluated on the test set.  
   - Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.  

---

## 📊 Key Outcomes / Results  

- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~95.9%  
- **Test Accuracy:** ~95%  

**Classification Report (Test Set):**  

- **Normal:** Precision = 0.93 | Recall = 0.98 | F1 = 0.96  
- **Pneumonia:** Precision = 0.98 | Recall = 0.95 | F1 = 0.96  
- **Tuberculosis:** Precision = 0.91 | Recall = 0.76 | F1 = 0.83  

**Insights:**  
- Strong performance for **Normal** and **Pneumonia**.  
- Tuberculosis classification slightly weaker due to dataset imbalance.  

---

## ✅ Conclusion  
This project demonstrates that a simple CNN can achieve **95% accuracy** in classifying chest X-rays into **Normal, Pneumonia, and Tuberculosis**.  
The results highlight the potential of deep learning in **medical imaging** and early disease detection.  
