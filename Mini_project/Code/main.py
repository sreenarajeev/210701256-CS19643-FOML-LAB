import streamlit as st
import tensorflow as tf
import numpy as np

def model_predict_image(image):
    model = tf.keras.models.load_model('trained_model.h5')
    img = tf.keras.preprocessing.image.load_img(image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Background image
original_title = '<h2 style="font-family:Poppins,sans-serif; color: #00008B; font-size: 40px;">Plant Palette</h2>'
st.markdown(original_title, unsafe_allow_html=True)
header = '<h3 style="font-family:Poppins,sans-serif; color: #337357; font-size: 35px;">A Fruit - Vegetable Classifier Model</h3>'
st.markdown(header, unsafe_allow_html=True)


image = st.file_uploader("Choose an Image")
button_col1, button_col2 = st.columns(2)

with button_col1:
    if st.button('Show Image') and image:
        st.image(image, width=400)

with button_col2:
    if st.button('Predict'):
        st.spinner()
        if image is not None:
            image_placeholder = st.empty()
            # st.image(image, width=400)
            result_index = model_predict_image(image)
            with open("labels.txt") as f:
              content = f.readlines()
            label = [i for i in content]
            image_placeholder.image(image, width=400)
            st.success("The given image is {}".format(label[result_index].capitalize()))

# Adding CSS to create space between buttons
st.markdown(
    """
    <style>
    .stButton > button {
        margin-top: 20px;
        margin-left: 140px;
        background-color: #7DB862 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)