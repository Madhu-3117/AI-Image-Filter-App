import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from filters import cartoonify, pencil_sketch, sepia, adjust_brightness_contrast, blur_faces

st.title("🎨 AI Image Filter App - Enhanced")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="📸 Original Image", use_column_width=True)

    filter_option = st.selectbox(
        "Choose a filter",
        ["None", "Cartoonify", "Pencil Sketch", "Sepia", "Blur Faces", "Adjust Brightness/Contrast"]
    )

    output_image = image_cv.copy()

    if filter_option == "Cartoonify":
        output_image = cartoonify(image_cv)
    elif filter_option == "Pencil Sketch":
        output_image = pencil_sketch(image_cv)
    elif filter_option == "Sepia":
        output_image = sepia(image_cv)
    elif filter_option == "Blur Faces":
        output_image = blur_faces(image_cv)
    elif filter_option == "Adjust Brightness/Contrast":
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
        output_image = adjust_brightness_contrast(image_cv, brightness, contrast)

    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="✨ Processed Image", use_column_width=True)

    # Save option
    save_path = f"outputs/processed.png"
    cv2.imwrite(save_path, output_image)
    st.success("✅ Image processed successfully!")
    with open(save_path, "rb") as file:
        st.download_button(label="📥 Download Image", data=file, file_name="filtered_image.png", mime="image/png")
