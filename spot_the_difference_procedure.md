# Spot the Difference ML Workflow: Step-by-Step Explanation

This document explains the procedure for detecting added, removed, or changed objects between two similar images using advanced machine learning techniques. The workflow combines data preparation, change localization, object detection, and matching to robustly spot differences.

---

## 1. Data Preparation & Vocabulary
- **Normalize object labels:** Clean and standardize object names in your dataset (e.g., 'man', 'guy', 'worker' → 'person').
- **Build vocabulary:** Extract a list of unique object types to detect (e.g., 'car', 'person', 'cone').

## 2. Change Localization (Where things changed)
- **Siamese backbone:** Use a twin neural network (e.g., ViT/Swin Transformer) to process both images in parallel, extracting features.
- **Cross-attention:** Compare features between images to focus on regions that differ.
- **Change logit map (H):** Output a multi-scale map highlighting areas where changes likely occurred.

## 3. Object Detection (What objects changed)
- **Open-vocabulary detector:** Use a model like OWL-ViT or Grounding DINO to detect objects in both images, using your vocabulary.
- **Bounding boxes & labels:** Get locations and types of objects present in each image.

## 4. Score Fusion
- **Combine scores:** Boost detector confidence for objects overlapping with high-change regions in the change map (H).
- **Formula:**
  
  $\text{score}' = \text{score}_{det} \times (1 + \lambda \times \text{normalized H overlap})$

## 5. Matching & Decision Rules
- **Match objects:** Use class labels and bounding box overlap (IoU) to match objects between images.
- **Rules:**
  - Only in second image → "added"
  - Only in first image → "removed"
  - Matched but moved/appearance changed → "changed"

## 6. Classification Heads (Optional)
- **Global features:** Add heads to predict, for each class, whether it was added, removed, or changed.
- **Weak supervision:** Train using category-level labels, not pixel-perfect masks.

## 7. Final Output
- **For each image pair:** Output lists of added, removed, and changed objects.
- **Visualization:** Display results and save in the required format for submission.

---

## How the Techniques Work Together
- **Siamese encoder & change map:** Tell you where to look for changes.
- **Detector:** Tells you what objects are present.
- **Matching & fusion:** Decide what changed and how (added, removed, changed).
- **Weak supervision & symmetry tricks:** Enable learning even with limited labels.

This pipeline allows robust difference detection between images, combining deep learning, object detection, and smart matching—even when only category-level labels are available.
