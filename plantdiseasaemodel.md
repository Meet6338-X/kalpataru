---
license: apache-2.0
language:
- en
metrics:
- accuracy
pipeline_tag: image-classification
tags:
- plant-disease
- mobilenetv2
- image-classification
- computer-vision
- deep-learning
- agriculture
- cnn
---

# 🌿 Plant Disease Classification – MobileNetV2

### **High-Accuracy Deep Learning Model for Crop Disease Detection**

**Author:** Daksh Goyal


## 🚀 Overview

This repository provides a **lightweight, production-ready MobileNetV2 model** trained on the **PlantVillage** augmented dataset for multi-class plant disease classification (38 classes).

The model is optimized for:

✔ Real-time inference
✔ Low-compute environments (mobile/edge devices)
✔ Web apps (Streamlit)
✔ Integration with farm-tech solutions


## 🧠 Model Summary

* **Architecture:** MobileNetV2
* **Dataset:** New Plant Diseases Dataset (Augmented) – PlantVillage
* **Total Classes:** **38**
* **Training Images:** **~87,000 images** (balanced across 38 classes)
* **Validation Accuracy:** **95%**
* **Loss:** Cross Entropy
* **Optimizer:** Adam (LR = 0.001)
* **Input Size:** **224 × 224**
* **Format:** PyTorch `.pth`


## 🌐 Live Demo

Try the fully deployed, interactive plant-disease detection app:

👉 **Streamlit App:**
**[https://agriguard271005.streamlit.app/](https://agriguard271005.streamlit.app/)**

Upload a leaf image → get:
✔ Disease name
✔ Confidence score
✔ Health status
✔ Severity level
✔ Recommended action


## 🤗 HuggingFace Model

The pretrained `.pth` model and configuration files are hosted here:

👉 **HuggingFace Model Repo:**
**[https://huggingface.co/Daksh159/plant-disease-mobilenetv2](https://huggingface.co/Daksh159/plant-disease-mobilenetv2)**

Available assets:

* `mobilenetv2_plant.pth`
* `class_names.json`
* `README.md` (this file)


## 🏗 Model Architecture (Training Setup)

### **Backbone: MobileNetV2**

```python
model = models.mobilenet_v2(pretrained=True)

for p in model.features.parameters():
    p.requires_grad = False  # freeze feature extractor

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 38)
)
```

### **Training Loop**

* 10 epochs
* Augmented inputs
* Adam optimizer
* Early stopping possible


## 📊 Supported Classes

Includes diseases & healthy leaves of:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato
* Orange

(Full list of 38 classes included in `class_names.json`)


## 🖼 Inference Example

```python
import torch
from torchvision import models, transforms
from PIL import Image

model = ...   # load your mobilenetv2_plant.pth
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

img = Image.open("leaf.jpg").convert("RGB")
x = tf(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)
    probs = torch.softmax(preds, dim=1)[0]
    index = probs.argmax().item()

print("Predicted Class:", CLASS_NAMES[index])
print("Confidence:", float(probs[index]))
```


## 🔌 API / Integration Use Cases

This model can be integrated into:

🌱 Farming mobile apps
🌾 Agri-tech dashboards
💡 Real-time disease monitoring systems
📸 Drone-based crop scanning pipelines
🌍 IoT/edge deployment for field devices


## 🖥 Deployment

### **Available Deployments**

| Platform            | Status         | Link                                                                                                                   |
| ------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **HuggingFace**     | ✔ Model Hosted | [https://huggingface.co/Daksh159/plant-disease-mobilenetv2](https://huggingface.co/Daksh159/plant-disease-mobilenetv2) |
| **Streamlit Cloud** | ✔ Web App Live | [https://agriguard271005.streamlit.app/](https://agriguard271005.streamlit.app/)                                       |


## 🏁 Future Improvements

Planned enhancements:

* Grad-CAM heatmap explainability
* Faster ONNX / TensorRT export
* Support for multiple leaves in a single image
* Region-based crop disease localization
* Larger dataset fine-tuning


## 🙌 Acknowledgements

* PlantVillage Dataset
* PyTorch
* Kaggle
* Streamlit Cloud
* HuggingFace 🤗