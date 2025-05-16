# MRI-based-Brain-Tumor-Segmentation
![Annotations](https://github.com/user-attachments/assets/106850f3-2718-4a78-86f4-60747246b189)

Brain tumor segmentation is a critical task in medical image analysis, aimed at identifying and delineating tumor regions from MRI scans. Accurate segmentation is essential for diagnosis, treatment planning, and monitoring of brain tumors. Manual segmentation is time-consuming and subject to inter-observer variability, making automated deep learning approaches increasingly important.
This project focuses on developing a deep learning model to automatically segment brain tumors using the BraTS2020 dataset. The aim is to classify and localize sub-regions such as enhancing tumor (ET), tumor core (TC), and whole tumor (WT) from multi-modal MRI scans.

## Project Overview
The BraTS 2020 dataset includes multi-modal MRI scans of patients with gliomas (both high-grade and low-grade). Each patient scan contains four modalities:

![sdasd](https://github.com/user-attachments/assets/c4cd0dcf-b45b-4235-be1c-f378f2636c77)

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

![qweqweqweq](https://github.com/user-attachments/assets/b57420cf-0573-4d8b-9a94-124886f075d2)

Volume Slice Selection
- Rather than feeding the entire 3D MRI volume into the network—which can be both memory-intensive and include many empty or non-tumor regions—we extract a fixed number of informative 2D slices from each scan:

Fixed Slice Count (VOLUME_SLICES = 100)
- Ensures that every training sample contributes the same number of 2D images, simplifying batching and stabilizing GPU memory usage.

Offset Start (VOLUME_START_AT = 22)
- Skips the initial slices (often containing little or no brain tissue) and focuses on the middle portion of the scan where tumors are most likely to appear.

Multi-Modal Inputs
- For each selected slice index j, we load both FLAIR and T1-CE images, resize them to (128 × 128), and stack them as two channels. This preserves the complementary contrast information critical for distinguishing tumor subregions.

Balanced Data Shape
- By extracting exactly 100 slices per case and then reshaping into a batch of (100 × height × width × 2), the generator produces uniformly-sized inputs (100, 128, 128, 2) and corresponding one-hot masks (100, 128, 128, 4).

Clinical Relevance
- Tumor tissue typically occupies central slices rather than the very top or bottom of the brain volume. Centering the slice window improves the chance of seeing tumor regions in every batch, which in turn accelerates learning and boosts segmentation accuracy.
![asdasdasda](https://github.com/user-attachments/assets/dafb0bb2-160a-406a-9ea9-dbd9e3ea654d)


## Dataset Path structure

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
│   │       │   ├── BraTS2020_Training_002_flair.nii  
│   │       │   ├── BraTS2020_Training_002_seg.nii  
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
│           │   ├── BraTS2020_Validation_002_flair.nii  
│           │   ├── BraTS2020_Validation_002_seg.nii  
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

## Conclusion

By the end of training, our U-Net felt like it had grown into a true expert tumor detective. Watching the loss curves steadily fall—and then hover comfortably around 0.03—gave me confidence that the model wasn’t just memorizing, but genuinely learning how to tell tumor from healthy tissue. Even more exciting was the Dice score climbing from the high 0.90s into the very top of the 0.99 club: in plain English, that means our predictions almost perfectly overlapped with the radiologist-annotated masks, slice after slice.

![outputawq](https://github.com/user-attachments/assets/16a0f868-2a85-46f1-b885-ab4c0113e75c)


But numbers alone don’t tell the whole story. When I overlay the model’s output on the original FLAIR image, you can actually see edema, necrotic core, and enhancing tumor pop out in different colors, hugging the true tumor boundaries so closely that it’s hard to tell prediction from ground truth at a glance. And in the clinic, missing a single tumor voxel can make all the difference—so seeing sensitivity soar above 99% felt like a mini triumph. Equally reassuring was specificity sitting just as high, showing our network wasn’t overzealous and accidentally painting healthy brain tissue as disease.

![output](https://github.com/user-attachments/assets/7801dcab-a869-4276-964f-393088b1a32c)


All this adds up to more than just impressive statistics: it means we’re one step closer to a tool that could help radiologists speed up their workflow, catch subtle tumor parts that might slip past human eyes, and ultimately guide more informed treatment decisions. Of course, there’s room to grow—3D context, attention mechanisms, and adapting to other MRI protocols are next on the wishlist—but these results give me real optimism. This U-Net really understands brain tumors, and that feels like a small but meaningful leap toward better, faster, and more consistent medical imaging.









