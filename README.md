# Custom Image Classifier with PyTorch & Tkinter

A sleek, desktop-based image classification application built using Python, PyTorch, and Tkinter. This application allows users to dynamically add custom classes, copy training images, train a linear classification head on top of a pre-trained ResNet-18 feature extractor, and run single or batch folder predictions.

---

## 🚀 Features

* **Pre-trained Backbone:** Uses a state-of-the-art ResNet-18 architecture as a rigid feature extractor for high-accuracy embeddings.
* **Dynamic Custom Head:** Add, manage, and remove custom image classes on the fly.
* **Asynchronous Processing:** Long-running model loads, training cycles, and evaluations run on background threads to keep the UI entirely responsive.
* **Modern UI/UX:** Dark-mode themed dashboard styled via `tkinter.ttk` with an integrated real-time activity log and smooth progress/ETA tracking.
* **Local Storage:** Automatically saves and loads your trained linear classification weights (`custom_head.pth`) and class mapping definitions (`custom_classes.json`).

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/custom-image-classifier.git](https://github.com/yourusername/custom-image-classifier.git)
cd custom-image-classifier
2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required external libraries using pip:

Bash
pip install torch torchvision pillow
Note: If you want GPU acceleration, make sure to install the CUDA-enabled version of PyTorch corresponding to your hardware.

3. Run the Application
Bash
python ImageClassificationCustomTraining.py
📂 Architecture & Workflow
The application leverages Transfer Learning. Instead of training a deep convolutional network from scratch, it freezes a pre-trained backbone and optimizes a single linear classification layer.

┌─────────────────────────────────┐
│     Input Image (224x224)       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       ResNet-18 Backbone        │  ◀── Weights Frozen (requires_grad=False)
└────────────────┬────────────────┘
                 │ (Extracts 512 Features)
                 ▼
┌─────────────────────────────────┐
│      Custom Linear Head         │  ◀── Trained on your custom dataset
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Predicted Class + Confidence   │
└─────────────────────────────────┘
Feature Extraction: Images are preprocessed and fed into a ResNet-18 model. The final fully connected (fc) layer is replaced with nn.Identity(), converting the model into a static 512-dimensional feature vector extractor.

Classification Head: A dynamic nn.Linear(512, num_classes) head is mapped dynamically to match whatever categories you insert via the UI.

📝 Pending Implementation Steps
The provided script contains scaffolding for complete application lifecycle management. To make the program fully operational, you must complete the following stub methods inside the CustomClassifier and ModernClassifierApp classes:

1. CustomClassifier.predict(self, tensor)
Goal: Feed the image tensor through the backbone to extract features, pass features to the head, apply a softmax layer, and return the predicted class index and confidence percentage.

2. CustomClassifier.train_model(self, log=None, progress=None)
Goal: Build a training pipeline using standard cross-entropy loss (nn.CrossEntropyLoss) and an optimizer (e.g., optim.SGD or optim.Adam). It should scan the custom_classes/ directory using PyTorch's Dataset or DataLoader, cycle through the configured number of epochs, and report updates back to the UI via the provided callbacks.

3. ModernClassifierApp._start_single(self)
Goal: Gather the file path from the input box, preprocess the single target image using preprocess_image(), trigger background execution to run model.predict(), and call _update_preview() and _set_info() to show the output.

4. ModernClassifierApp._start_folder(self)
Goal: Batch process all images matching valid extensions inside the targeted folder, displaying global percentage progress bars alongside real-time ETA calculations.

5. ModernClassifierApp._start_training(self)
Goal: Thread the wrapper handler for model.train_model(), change UI element modes to busy during execution via _set_busy(True), and trigger updates upon training completion.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
