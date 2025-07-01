# Ai-Animal-calsification – Local Model Training Documentation
Welcome to the documentation for the AI Animal Image Classifier project. This guide walks through how the model was trained and tested on a local machine using PyTorch.
# Project Overview
This project focuses on building a deep learning model that classifies animal images into one of several predefined species. The MobileNetV2 architecture was used due to its balance of speed and accuracy, ideal for lightweight real-time predictions.

# Training Configuration
- **Model Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Dataset:** Custom animal image dataset with 15 classes
- **Trained Animal Classes:**
    ```Bear
    Bird
    Cat
    Cow
    Deer
    Dog
    Dolphin
    Elephant
    Giraffe
    Horse
    Kangaroo
    Lion
    Panda
    Tiger
    Zebra
    ```

- **Loss Function:** CrossEntropyLoss

- **Optimizer:** Adam (assumed)

- **Transformations:** Resize, ToTensor, Normalize

# Training Progress Per Epoch
- **Epochs:** 10
- **Best Accuracy Achieved:** Epoch 7 (97.89%)
    ```
    ✅ Epoch 1/10: Loss=94.7497, Accuracy=77.62%
    ✅ Epoch 2/10: Loss=43.4557, Accuracy=89.66%
    ✅ Epoch 3/10: Loss=29.6786, Accuracy=93.26%
    ✅ Epoch 4/10: Loss=25.4574, Accuracy=93.42%
    ✅ Epoch 5/10: Loss=18.8664, Accuracy=95.11%
    ✅ Epoch 6/10: Loss=17.8976, Accuracy=95.68%
    ✅ Epoch 7/10: Loss=10.9166, Accuracy=97.89%
    ✅ Epoch 8/10: Loss=15.7553, Accuracy=95.94%
    ✅ Epoch 9/10: Loss=17.3483, Accuracy=95.88%
    ✅ Epoch 10/10: Loss=15.6635, Accuracy=95.99%
    ```

# Key Insights

- The model showed strong learning progression from the very first epoch.
- Achieved >95% accuracy from Epoch 5 onwards.
- MobileNetV2 proved ideal for high accuracy with efficient resource usage.
- Suitable for future integration with lightweight deployment environments or edge devices.

# Show imgae & Prediction (Optional)
- This is optional — you can skip this code block if visual display is not needed.
``` ## Show image
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {result}", fontsize=16)
    plt.axis('off')
    plt.show()
```