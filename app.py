import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

model = load_model("mnist_model.h5")

st.title("Handwritten Digit Recognizer")

canvas = st.canvas(draw_mode="freedraw", stroke_width=15, stroke_color='#FFFFFF',
                   background_color='#000000', height=280, width=280)

if canvas.image_data is not None:
    img = Image.fromarray((canvas.image_data[:, :, 0:3]).astype('uint8'), 'RGB')
    img = img.convert('L')  # grayscale
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
