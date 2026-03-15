

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import urllib.request


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector(nn.Module):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer for 7 emotions
        self.model.fc = nn.Linear(512, 7)

    def forward(self, x):
        return self.model(x)

def download_pretrained_model():
    """
    Download or create a basic emotion detection model
    """
    model_path = 'emotion_model.pth'

    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return True

    print("Creating emotion detection model...")
    print("Note: This is a basic model. For better accuracy, train on FER2013 dataset.")

    try:
        # Create and save a basic model with random weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EmotionDetector()
        model = model.to(device)

        # Save the model state dict
        torch.save(model.state_dict(), model_path)
        print(f"✓ Basic model created: {model_path}")
        print("⚠️  Note: This model has random weights. For production use,")
        print("   consider training on the FER2013 dataset or using a pre-trained model.")
        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def verify_model():
    """
    Verify the emotion detection model can be loaded and used
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EmotionDetector()
        model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
        model = model.to(device)
        model.eval()

        print("\n✓ Model verification successful!")
        print(f"Model device: {device}")
        print(f"Expected emotions: {EMOTION_LABELS}")

        # Test with random input
        test_image = torch.randn(1, 3, 48, 48).to(device)  # RGB input for ResNet
        with torch.no_grad():
            predictions = model(test_image)
            probabilities = torch.softmax(predictions, dim=1)
            print(f"Test prediction shape: {predictions.shape}")
            print(f"Test probabilities shape: {probabilities.shape}")

        print("✓ Model is ready to use!")
        return True

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False
        
       
if __name__ == "__main__":
    print("="*60)
    print("EMOTION DETECTION MODEL SETUP (PyTorch)")
    print("="*60)


    success = download_pretrained_model()

    if success:

        verify_model()

    print("\n" + "="*60)
    print("Setup complete! You can now run: python app.py")
    print("="*60)