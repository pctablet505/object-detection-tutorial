# Object Detection as a Machine Learning Problem

> 🎓 An interactive, beginner-friendly notebook that deconstructs object detection into its core machine learning building blocks.

---

## Learning Objectives

By working through this repository you will understand:

1. **How object detection is framed as a regression + classification problem**
2. **What bounding box regression means** and how a model learns to predict `(x1, y1, x2, y2)`
3. **Binary classification** for determining whether an object exists in a proposed region
4. **Multi-class classification** for identifying the type of object inside a bounding box
5. **How synthetic data can be used** to prototype and debug detection pipelines before scaling to real-world datasets

---

## Concepts Covered

| Topic | Description |
|-------|-------------|
| **Bounding Box Regression** | Predicting four continuous coordinates that localize an object |
| **Binary Classification** | Objectness score — does this region contain an object? |
| **Multi-class Classification** | Class label prediction (e.g., circle, rectangle, triangle) |
| **IoU (Intersection over Union)** | Evaluating overlap between predicted and ground-truth boxes |
| **Synthetic Dataset Generation** | Programmatically creating shapes with known labels for controlled experiments |

---

## File Structure

```
object-detection-tutorial/
├── object_detection_ml_demo.ipynb   📓 Main interactive notebook
├── dataset_generation.py            🎲 Synthetic shape generator
├── hello.py                         👋 Minimal sanity-check script
├── images/
│   ├── background_foreground_head.png
│   ├── bounding_box_as_corner_regression.png
│   ├── classification_head.png
│   ├── iou_explained.png
│   └── multiple_object_detection.png
└── input-output/
    ├── input0.png
    ├── input1.png
    ├── output0.png
    └── output1.png
```

---

## How to Run the Notebook

### 1. Clone the repository
```bash
git clone https://github.com/pctablet505/object-detection-tutorial.git
cd object-detection-tutorial
```

### 2. Install dependencies
```bash
pip install opencv-python numpy matplotlib
```

### 3. Launch Jupyter
```bash
jupyter notebook object_detection_ml_demo.ipynb
```

> **No GPU required!** All examples run comfortably on CPU using small synthetic images.

---

## Sample Outputs Description

The `input-output/` folder contains side-by-side comparisons of:

- **`input0.png` / `input1.png`** — Synthetic scenes with colored shapes on random backgrounds
- **`output0.png` / `output1.png`** — The same scenes with predicted bounding boxes and class labels overlaid

These samples demonstrate the end-to-end pipeline: *generate data → train concepts → visualize predictions*.

---

## Who Is This For?

- Students and practitioners transitioning from image classification to object detection
- Anyone who wants to **see the math and code behind bounding boxes** before jumping into heavy frameworks like YOLO or Detectron2
- Educators looking for a self-contained, visual teaching resource

---

## Next Steps

Once you are comfortable with the fundamentals here, you can move on to:
- Modern one-stage detectors (YOLO, SSD)
- Two-stage detectors (Faster R-CNN)
- Transformer-based detectors (DETR, Deformable DETR)

Happy learning! 🚀
