# Image-to-Image Retrieval Using Vision Transformers and FAISS

## Objective

This project presents a complete pipeline for image-to-image retrieval using Vision Transformer (ViT)-based models and FAISS (Facebook AI Similarity Search) for fast similarity search. The pipeline is evaluated on the CIFAR-100 dataset and supports both dataset-based and external (custom) image queries.

---

## Progress Overview

Key milestones achieved:

* Implemented multiple state-of-the-art pretrained models.
* Unified feature extraction and similarity evaluation pipeline.
* FAISS integration for efficient nearest neighbor search using L2 distance.
* Tested retrieval for both known dataset queries and custom external images.
* Applied standardized performance metrics across all models.

---

## Model Selection

ViT-based models were selected due to:

* Global feature learning via self-attention mechanisms.
* Strong performance on vision benchmarks.
* Better generalization and patch-wise processing ideal for small datasets like CIFAR-100.

### Vision Transformer-Based Models Implemented:

* EVA-02
* CLIP (ViT-B/32)
* DINO (ViT-S)
* ViT (Base patch16)
* Swin Transformer (Hierarchical ViT variant)

### CNN-Based Baseline Models:

* ResNet-18
* ResNet-50
* EfficientNet-B3
* MobileNetV3
* DenseNet
* ConvNeXt
* RegNetX\_032
* RegNetY\_042

---

## Similarity Search with FAISS

FAISS was chosen for its efficiency and precision in handling vector similarity search tasks.

### Advantages:

* Supports exact nearest neighbor search via L2 distance.
* Scales well with moderate-sized datasets like CIFAR-100.
* Provides fast retrieval with low latency.
* Easy integration with PyTorch/Numpy workflows.

### Alternatives Considered:

| Technique | Reason for Rejection                                                             |
| --------- | -------------------------------------------------------------------------------- |
| LSH       | Hash collisions lead to lower retrieval accuracy.                                |
| HNSW      | High overhead not justified for small-scale datasets.                            |
| ScaNN     | Optimized for massive-scale retrieval; complex setup.                            |
| ChromaDB  | Built for multi-modal data; unnecessary for single-modal (image-only) use cases. |

---

## Evaluation Metrics

We used the following metrics for model evaluation:

* Precision\@K
* False Positive Rate (FPR)

For dataset-based queries, labels are compared with the top-K retrieved images. For custom queries, the label of the top-1 result is assumed to be correct.

Example:
If 6 out of 10 retrieved images match the label → Precision\@10 = 60%

---

## Pipeline Code Snippets

### 1. Installation and Setup

```python
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import faiss
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True).to(device).eval()
```

### 2. Preprocessing and Dataset Loading

```python
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_cifar100():
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    return dataset, dataloader
```

### 3. Feature Embedding Extraction

```python
def get_embeddings(dataloader, model):
    all_features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = model.forward_features(images).mean(dim=1)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)
```

### 4. FAISS Indexing and Search

```python
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def faiss_search(index, query_embedding, k=50):
    _, indices = index.search(query_embedding, k)
    return indices[0]
```

### 5. Metric Computation

```python
def compute_metrics(retrieved_indices, query_labels, database_labels, k):
    total_true_positives = 0
    total_false_positives = 0
    for i in range(len(query_labels)):
        retrieved = retrieved_indices[i]
        retrieved_labels = database_labels[retrieved]
        tp = np.sum(retrieved_labels == query_labels[i])
        fp = k - tp
        total_true_positives += tp
        total_false_positives += fp
    precision = total_true_positives / (len(query_labels) * k)
    fpr = total_false_positives / (len(query_labels) * k)
    return precision, fpr
```

---

## Custom Image Retrieval

The pipeline supports external (non-CIFAR) custom images.

Steps:

1. Load external image with PIL.
2. Apply the same preprocessing.
3. Generate embedding and retrieve using FAISS.

Example:

```python
from PIL import Image

img = Image.open("custom_image.jpg")
img_tensor = preprocess(img).unsqueeze(0).to(device)
query_feature = model.forward_features(img_tensor).mean(dim=1)
query_feature /= query_feature.norm(dim=-1, keepdim=True)
results = faiss_search(index, query_feature.cpu().numpy(), k=10)
```

---

## Visualization and Analysis

Line plots illustrate average Top-K Precision across 5 query samples.

### Insights:

* EVA-02, ConvNeXt, and EfficientNet-B3 maintain high accuracy across all K values.
* Swin Transformer and ResNet-18 show drops at higher K values, indicating less robustness.
![image](https://github.com/user-attachments/assets/cccba628-04af-473b-b4e1-3368d1c68731)

---

## 📚 References

* FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* timm: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
* CLIP: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
* DINO: [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)
* CIFAR-100: [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
