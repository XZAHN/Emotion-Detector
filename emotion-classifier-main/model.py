

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import os
import urllib.request
import gdown


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def download_pretrained_model():
    
    model_path = 'emotion_model.h5'
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return True
    
    print("Downloading pre-trained emotion detection model...")
    print("This may take a few minutes...")
    
    try:
        
        url = "https://github.com/atulapra/Emotion-detection/raw/master/model.h5"
        
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Model downloaded successfully: {model_path}")
        return True
        
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://github.com/atulapra/Emotion-detection")
        print("2. Download 'model.h5' file")
        print("3. Rename it to 'emotion_model.h5'")
        print("4. Place it in the project root directory")
        return False

def verify_model():
    
    try:
        model = load_model('emotion_model.h5', compile=False)
        print("\n✓ Model verification successful!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Expected emotions: {EMOTION_LABELS}")
        
       
        test_image = np.random.rand(1, 48, 48, 1)
        predictions = model.predict(test_image, verbose=0)
        print(f"\nTest prediction shape: {predictions.shape}")
        print("✓ Model is ready to use!")
        
        return True
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("EMOTION DETECTION MODEL SETUP")
    print("="*60)
    
    
    success = download_pretrained_model()
    
    if success:
        
        verify_model()
    
    print("\n" + "="*60)
    print("Setup complete! You can now run: python app.py")
    print("="*60)