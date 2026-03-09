import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_cancer_model.keras")

# Class labels
classes = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Vascular Lesion"
]

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        predictions = tf.squeeze(predictions)

        class_idx = tf.argmax(predictions)

        loss = predictions[class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.reduce_max(heatmap)

    return heatmap.numpy()


# Streamlit UI
st.title("Skin Disease Detection using Deep Learning")

uploaded_file = st.file_uploader("Upload a Skin Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_resized = image.resize((224,224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # check if image looks like dermoscopic skin image
    if np.std(img_array) < 0.05:
       st.warning("Invalid skin lesion image. Please upload a close-up skin lesion image.")
       st.stop()

    # Prediction
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = classes[predicted_index]

    # Threshold check
    if confidence < 97:
        st.warning("No Skin Disease Detected")
    else:

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Grad-CAM only if disease detected
        heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")

        heatmap = cv2.resize(heatmap,(224,224))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(
            np.array(image_resized), 0.6,
            heatmap, 0.4,
            0
        )

        st.image(
            superimposed_img.astype("uint8"),
            caption="Grad-CAM Visualization",
            use_container_width=True
        )

# Disclaimer
