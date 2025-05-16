# MRI-based-Brain-Tumor-Segmentation
![Annotations](https://github.com/user-attachments/assets/106850f3-2718-4a78-86f4-60747246b189)

Brain tumor segmentation is a critical task in medical image analysis, aimed at identifying and delineating tumor regions from MRI scans. Accurate segmentation is essential for diagnosis, treatment planning, and monitoring of brain tumors. Manual segmentation is time-consuming and subject to inter-observer variability, making automated deep learning approaches increasingly important.
This project focuses on developing a deep learning model to automatically segment brain tumors using the BraTS2020 dataset. The aim is to classify and localize sub-regions such as enhancing tumor (ET), tumor core (TC), and whole tumor (WT) from multi-modal MRI scans.

## Dataset Description
The BraTS 2020 dataset includes multi-modal MRI scans of patients with gliomas (both high-grade and low-grade). Each patient scan contains four modalities:

T1 – T1-weighted image
T1Gd – T1-weighted with contrast (gadolinium)
T2 – T2-weighted image
FLAIR – Fluid-attenuated inversion recovery
In brain tumor segmentation, the BraTS challenge divides tumors into three primary sub‑regions—necrotic/non‑enhancing core (NCR/NET), edema (ED), and enhancing tumor (ET)—and evaluates the “whole tumor” (WT) as their union. Each region reflects distinct pathology: the NCR/NET marks dead or non‑vascularized tissue, ED corresponds to fluid accumulation from blood–brain barrier disruption, and ET indicates active, vascularized tumor growth. Understanding these classes aids in diagnosis, treatment planning, and prognosis assessment. Tumor regions are labeled as:

Label 0: Background / Healthy Tissue
Label 1: Necrotic / Non-enhancing core (NCR/NET)
Label 2: Edema (ED)
Label 3: Enhancing tumor (ET)
Each tumor sub-region reflects different pathological conditions. The union of all three is referred to as the Whole Tumor (WT). Understanding these distinctions supports better clinical outcomes.

/dataset/
├── brats20-dataset-training-validation/
│   ├── BraTS2020_TrainingData/
│   │   └── MICCAI_BraT2020_TrainingData/
│   │       ├── BraTS2020_Training_001/
│   │       │   ├── BraTS2020_Training_001_flair.nii
│   │       │   ├── BraTS2020_Training_001_seg.nii
│   │       │   ├── BraTS2020_Training_001_t1.nii
│   │       │   ├── BraTS2020_Training_001_t1ce.nii
│   │       │   └── BraTS2020_Training_001_t2.nii
│   │       ├── BraTS2020_Training_002/
│   │       │   └── …
│   │       └── …
│   └── BraTS2020_ValidationData/
│       └── MICCAI_BraT2020_ValidationData/
│           ├── BraTS2020_Validation_001/
│           │   ├── BraTS2020_Validation_001_flair.nii
│           │   ├── BraTS2020_Validation_001_seg.nii
│           │   ├── BraTS2020_Validation_001_t1.nii
│           │   ├── BraTS2020_Validation_001_t1ce.nii
│           │   └── BraTS2020_Validation_001_t2.nii
│           ├── BraTS2020_Validation_002/
│           │   └── …
│           └── …


## Model Architecture and Evaluation Metrics

U-Net 

Strong Localization + Context:
- U-Net’s symmetric encoder–decoder with skip-connections excels at preserving fine spatial details (tumor boundaries) while   capturing high-level context (tumor heterogeneity). In BraTS, where small enhancing tumor regions matter, those skip-        connections help localize tiny lesions without losing global appearance.

Proven in Medical Imaging:
- Since its introduction, U-Net has become the de-facto standard for biomedical segmentation tasks. Its design works well      even with limited annotated data—common in medical settings—thanks to efficient feature reuse and data-augmentation          compatibility.

Flexible Multi-Modal Input:
- The BraTS dataset provides multiple MRI sequences per patient. U-Net seamlessly incorporates multi-channel inputs (e.g.      T1, T2, FLAIR) and outputs voxel-wise class probabilities for each tumor subregion, making it ideal for the four-class       (background + three tumor types) problem.

Dice-Based & Clinical-Oriented Metrics

Dice Coefficient & Dice Loss:
- Overlap-Focused: Dice directly measures voxel overlap, so optimizing Dice loss maximizes segmentation agreement even for small, irregular tumor regions.

- Class Imbalance Handling: Tumors often occupy a small fraction of brain volume; Dice loss naturally down-weights large background regions.

Combined Categorical Dice + Cross-Entropy:

- Stability + Precision: Cross-entropy enforces per-voxel correctness, while Dice encourages overall shape accuracy. Together, they yield both sharp boundaries and reliable class probabilities.

Sensitivity (Recall):

- Minimize Missed Lesions: In clinical practice, failing to detect tumor tissue (“false negatives”) can have severe consequences. Sensitivity quantifies true-positive rate, ensuring the model learns to catch as many tumor voxels as possible.

Specificity:

- Avoid Over-Segmentation: Equally important is not labeling healthy tissue as tumor (“false positives”), which could lead to unnecessary interventions. Specificity measures true-negative rate, keeping the model conservative where it should.






