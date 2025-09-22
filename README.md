# BrainTumorDetection

🧠 Brain Tumor Classification with MRI Images

This project uses the Brain Tumor Classification MRI dataset
 to build a deep learning pipeline for classifying brain tumors into four categories:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

📂 Dataset

The dataset consists of MRI scans split into Training and Testing sets with four subfolders:

brain_tumor_dataset/
│
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
│
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/

⚙️ Preprocessing Pipeline

Data Loading

Load images from training and testing folders.

Resize all images to 256x256 pixels for uniformity.

CLAHE Enhancement

Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve MRI contrast.

Especially useful for medical images where tumors can be subtle.

🔍 Example:

Original	CLAHE Enhanced

	

Label Encoding

Convert labels (glioma_tumor, etc.) into one-hot vectors for classification.

Train / Validation / Test Split

Use Kaggle’s Training/Testing split.

Further split training into training + validation.

Data Augmentation

Apply transformations like rotation, zoom, and shifts using ImageDataGenerator.

Prevents overfitting by increasing dataset diversity.

Normalization

Scale pixel values from [0, 255] → [0, 1].

Required for stable deep learning training.

Feature Extraction (VGG16)

Use pretrained VGG16 (ImageNet) as a feature extractor.

Extracts deep features from MRI scans without training from scratch.

Classifier Training (Logistic Regression)

Train a simple logistic regression model on extracted features.

Faster and requires less GPU power.

Evaluation

Generate classification report (accuracy, precision, recall, F1).

Plot confusion matrix for class-wise performance.

📊 Results

Metrics: Accuracy, Precision, Recall, F1-score.

Confusion Matrix shows class-level performance.

🚀 How to Run

Clone the repo and download the dataset from Kaggle:

git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification


Mount Google Drive in Colab or place dataset under ./brain_tumor_dataset/.

Install dependencies:

pip install tensorflow opencv-python matplotlib seaborn mplcyberpunk scikit-learn


Run the notebook step by step.

📌 Notes

CLAHE is applied before resizing to preserve local details.

Augmentation is applied only to training data (never to validation/test → avoids leakage).

Normalization ensures compatibility with CNN/VGG16 inputs.

VGG16 + Logistic Regression is lightweight. For end-to-end CNN training, replace logistic regression with custom CNN layers.

📚 References

Dataset: Kaggle – Brain Tumor Classification MRI

CLAHE in OpenCV

VGG16 Paper
