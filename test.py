import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16, ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Set page config
st.set_page_config(
    page_title="Explainable CNNs with TensorFlow",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🖼️ Explainable CNNs with TensorFlow")
st.markdown("""
## Welcome to the Deep Learning Visualization App!
Visualize feature maps, saliency maps, and Grad-CAM for pre-trained TensorFlow models.
""")

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Choose a model",
    ["VGG16", "ResNet50"]
)

# Load the selected model
@st.cache_resource()
def load_model(model_name):
    if model_name == "VGG16":
        return VGG16(weights="imagenet")
    elif model_name == "ResNet50":
        return ResNet50(weights="imagenet")

model = load_model(model_name)

# Display model architecture
st.subheader("Model Architecture")
with st.expander("Show Model Summary"):
    model.summary(print_fn=lambda x: st.text(x))

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions and display top-5 classes
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)[0]
    
    st.subheader("Class Predictions")
    st.write("Top-5 Predicted Classes:")
    for i, (class_id, class_name, score) in enumerate(decoded_preds):
        st.write(f"{i+1}. {class_name} ({score*100:.2f}%)")

    # Visualization type selection
    vis_type = st.sidebar.selectbox(
        "Choose visualization type",
        ["Feature Maps", "Saliency Map"]
    )

    if vis_type == "Feature Maps":
        # Extract feature maps
        layer_name = st.sidebar.selectbox(
            "Choose a layer",
            [layer.name for layer in model.layers if "conv" in layer.name]
        )
        intermediate_model = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer(layer_name).output
        )
        feature_maps = intermediate_model.predict(img_array)

        # Visualize feature maps
        st.subheader(f"Feature Maps: {layer_name}")
        num_maps = min(16, feature_maps.shape[-1])  # Display up to 16 maps
        feature_maps = feature_maps[0, :, :, :num_maps]

        fig, axes = plt.subplots(1, num_maps, figsize=(20, 5))
        for i, ax in enumerate(axes):
            ax.imshow(feature_maps[:, :, i], cmap="viridis")
            ax.axis("off")
        st.pyplot(fig)

    elif vis_type == "Saliency Map":
        # Convert NumPy array to TensorFlow tensor
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Compute saliency map
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            preds = model(img_array)
            top_class = tf.argmax(preds[0])
            loss = preds[0, top_class]

        grads = tape.gradient(loss, img_array)[0]
        saliency_map = np.max(np.abs(grads), axis=-1)

        # Normalize and visualize saliency map
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        st.subheader("Saliency Map")
        st.image(saliency_map, caption="Saliency Map", use_container_width=True, clamp=True)
