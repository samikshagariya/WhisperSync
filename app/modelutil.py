# modelutil.py
from model import build_model
import os

def load_model():
    model = build_model()
    model.load_weights(os.path.join('models', 'fine_tuned_on_s2_demo.weights.h5'))
    return model
