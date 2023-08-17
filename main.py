import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Loading the trained model
trained_model = tf.keras.models.load_model("trained_model")

# face mask prediction function
def mask_prediction(path, model):
    img = Image.open(path)
    img = img.resize((128,128))
    img = img.convert("RGB")
    img = np.array(img).reshape(1,128,128,3)
    img = img/255
    y_pred = model.predict(img)

    if y_pred[0][0] >= 0.5:
        return "This person is wearing a mask!"
    else:
        return "This person isn't wearing a mask!"

# The main function
def main():
    st.title("Face mask detection system")
    image = st.file_uploader("Upload your image here!")
    if image:
        st.image(image, caption="Uploaded image")
        result = mask_prediction(image, trained_model)
        st.success(result)


if __name__ == "__main__":
    main()