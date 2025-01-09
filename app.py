import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import requests
from torchinfo import summary
from torchviz import make_dot

# Set page config
st.set_page_config(
    page_title="Explainable CNNs",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🖼️ Explainable CNNs")
st.markdown("""
## Welcome to the Deep Learning Visualization App!
This app allows you to visualize feature maps, saliency maps, Grad-CAM, and feature visualizations for various pre-trained models.
- **Step 1**: Choose a model from the sidebar.
- **Step 2**: Select the visualization type.
- **Step 3**: Upload an image (if required).
- **Step 4**: Explore the results!
""")

# Function to load ImageNet class labels
def load_imagenet_labels():
    # Download the ImageNet class labels
    labels_url = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
    response = requests.get(labels_url)
    labels = response.text.splitlines()
    return labels

# Load ImageNet class labels
imagenet_labels = load_imagenet_labels()

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "VGG11", "VGG13", "VGG16", "VGG19",
        "AlexNet",
        "DenseNet121", "DenseNet169", "DenseNet201",
        "InceptionV3",
        "MobileNetV2",
        "SqueezeNet1_0", "SqueezeNet1_1",
        "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0"
    ]
)

# Load the selected model
@st.cache_resource()
def load_model(model_name):
    if model_name == "ResNet18":
        return models.resnet18(pretrained=True)
    elif model_name == "ResNet34":
        return models.resnet34(pretrained=True)
    elif model_name == "ResNet50":
        return models.resnet50(pretrained=True)
    elif model_name == "ResNet101":
        return models.resnet101(pretrained=True)
    elif model_name == "ResNet152":
        return models.resnet152(pretrained=True)
    elif model_name == "VGG11":
        return models.vgg11(pretrained=True)
    elif model_name == "VGG13":
        return models.vgg13(pretrained=True)
    elif model_name == "VGG16":
        return models.vgg16(pretrained=True)
    elif model_name == "VGG19":
        return models.vgg19(pretrained=True)
    elif model_name == "AlexNet":
        return models.alexnet(pretrained=True)
    elif model_name == "DenseNet121":
        return models.densenet121(pretrained=True)
    elif model_name == "DenseNet169":
        return models.densenet169(pretrained=True)
    elif model_name == "DenseNet201":
        return models.densenet201(pretrained=True)
    elif model_name == "InceptionV3":
        return models.inception_v3(pretrained=True)
    elif model_name == "MobileNetV2":
        return models.mobilenet_v2(pretrained=True)
    elif model_name == "SqueezeNet1_0":
        return models.squeezenet1_0(pretrained=True)
    elif model_name == "SqueezeNet1_1":
        return models.squeezenet1_1(pretrained=True)
    elif model_name == "ShuffleNetV2_x0_5":
        return models.shufflenet_v2_x0_5(pretrained=True)
    elif model_name == "ShuffleNetV2_x1_0":
        return models.shufflenet_v2_x1_0(pretrained=True)

model = load_model(model_name)

# Display the model architecture
st.subheader("Model Architecture")
with st.expander("Show Model Summary"):
    # Generate model summary and write to the buffer
    model_stats = summary(
        model, 
        input_size=(1, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params"], 
        verbose=0, 
    )
    summary_str = str(model_stats)
    st.text(summary_str)
    
# dummy input
example_input = torch.randn(1, 3, 224, 224)

# Generate the visualization
dot = make_dot(model(example_input), params=dict(model.named_parameters()))
dot.format = "png"  # Output format
dot.render("model_architecture")  # Save to file


st.subheader("Model Architecture Diagram")
with st.expander("Show Model Architecture"):
    # Display in Streamlit
    st.image("model_architecture.png")

model.eval()  # Set the model to evaluation mode
# Visualization type selection
vis_type = st.sidebar.selectbox(
    "Choose visualization type",
    ["Feature Maps", "Saliency Map", "Grad-CAM", "Feature Visualization by Optimization"]
)

# Layer selection (for feature maps, Grad-CAM, and feature visualization by optimization)
if vis_type in ["Feature Maps", "Grad-CAM", "Feature Visualization by Optimization"]:
    layer_names = list(model.named_children())
    layer_name = st.sidebar.selectbox(
        "Choose a layer",
        [name for name, _ in layer_names]
    )

# Hook to capture feature maps (for feature maps, Grad-CAM, and feature visualization by optimization)
feature_maps = []
gradients = []

def hook_fn(module, input, output):
    feature_maps.append(output)

def grad_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

if vis_type in ["Feature Maps", "Grad-CAM", "Feature Visualization by Optimization"]:
    selected_layer = dict(layer_names)[layer_name]
    hook = selected_layer.register_forward_hook(hook_fn)
    grad_hook = selected_layer.register_backward_hook(grad_hook)

# Image upload (not needed for feature visualization by optimization)
if vis_type != "Feature Visualization by Optimization":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move the input and model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_batch = input_batch.to(device)
        model = model.to(device)

        if vis_type == "Feature Maps":
            # Forward pass through the model
            with torch.no_grad():
                model(input_batch)

            # Remove the hook
            hook.remove()

            # Visualize the feature maps
            def visualize_feature_maps(feature_maps, num_maps=16):
                # Get the feature maps from the hook
                maps = feature_maps[0].cpu()

                # Select a subset of feature maps to visualize
                maps = maps[0, :num_maps, :, :]  # Select the first `num_maps` channels

                # Normalize the feature maps for visualization
                maps = (maps - maps.min()) / (maps.max() - maps.min())

                # Create a grid of feature maps
                grid = make_grid(maps.unsqueeze(1), nrow=4, padding=2, normalize=False)

                # Convert the grid to a numpy array and rearrange dimensions for matplotlib
                grid_np = grid.numpy().transpose((1, 2, 0))

                # Display the grid
                st.image(grid_np, caption=f"Feature Maps from {layer_name}", use_container_width=True)

            visualize_feature_maps(feature_maps)

        elif vis_type == "Saliency Map":
            # Enable gradients for the input image
            input_batch.requires_grad = True

            # Forward pass
            output = model(input_batch)

            # Get the predicted class
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()

            # Display the predicted class
            st.write(f"Predicted Class: {imagenet_labels[predicted_class]}")

            # Compute the gradient of the output with respect to the input
            output[:, predicted_class].backward()

            # Get the saliency map
            saliency_map = input_batch.grad.data.abs().max(dim=1)[0].squeeze().cpu()

            # Normalize the saliency map
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            # Convert the saliency map to a numpy array
            saliency_map_np = saliency_map.numpy()

            # Display the saliency map
            st.image(saliency_map_np, caption="Saliency Map", use_container_width=True, clamp=True)

        elif vis_type == "Grad-CAM":
            # Enable gradients for the input image
            input_batch.requires_grad = True

            # Forward pass
            output = model(input_batch)

            # Get the predicted class
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()

            # Display the predicted class
            st.write(f"Predicted Class: {imagenet_labels[predicted_class]}")

            # Compute the gradient of the output with respect to the feature maps
            output[:, predicted_class].backward()

            # Get the gradients and feature maps
            grads = gradients[0].cpu().data.numpy().squeeze()
            fmap = feature_maps[0].cpu().data.numpy().squeeze()

            # Compute the weights for Grad-CAM
            weights = np.mean(grads, axis=(1, 2))

            # Compute the Grad-CAM heatmap
            grad_cam = np.zeros(fmap.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                grad_cam += w * fmap[i, :, :]

            # Apply ReLU to the Grad-CAM heatmap
            grad_cam = np.maximum(grad_cam, 0)

            # Normalize the Grad-CAM heatmap
            grad_cam = grad_cam / grad_cam.max()

            # Resize the Grad-CAM heatmap to match the input image size
            grad_cam = cv2.resize(grad_cam, (224, 224))

            # Convert the heatmap to a color map
            heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

            # Superimpose the heatmap on the original image
            input_image = input_tensor.numpy().transpose((1, 2, 0))
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image = np.uint8(255 * input_image)
            superimposed_img = cv2.addWeighted(input_image, 0.5, heatmap, 0.5, 0)

            # Display the Grad-CAM heatmap and superimposed image
            st.image(grad_cam, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
            st.image(superimposed_img, caption="Superimposed Grad-CAM", use_container_width=True, clamp=True)

elif vis_type == "Feature Visualization by Optimization":
    # Parameters for feature visualization by optimization
    st.sidebar.subheader("Optimization Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    num_iterations = st.sidebar.slider("Number of Iterations", 100, 1000, 200)
    regularization = st.sidebar.slider("Regularization Strength", 0.0, 0.1, 0.01)

    # Create a random noisy image
    input_image = torch.randn(1, 3, 224, 224, requires_grad=True)  # Random image with gradients
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = input_image.to(device)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam([input_image], lr=learning_rate)

    # Progress bar
    progress_bar = st.progress(0)

    # Optimization loop
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        feature_maps.clear()  # Clear previous activations
        model(input_image)

        # Get the activation of the target layer
        target_activation = feature_maps[0].mean()

        # Apply L2 regularization to the image
        l2_reg = regularization * torch.norm(input_image)

        # Loss function: Maximize the target activation while minimizing regularization
        loss = -target_activation + l2_reg

        # Backward pass and optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        # Clamp the image values to a valid range (out-of-place operation)
        with torch.no_grad():
            input_image.data = torch.clamp(input_image.data, -2, 2)

        # Update progress bar
        progress_bar.progress((i + 1) / num_iterations)

    # Remove the hook
    hook.remove()

    # Convert the optimized image to a displayable format
    def denormalize_image(image):
        # Reverse the normalization used in pre-trained models
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        return image * std + mean

    optimized_image = denormalize_image(input_image).squeeze(0).cpu().detach()

    # Convert to numpy and rearrange dimensions for matplotlib
    optimized_image = optimized_image.permute(1, 2, 0).numpy()
    optimized_image = np.clip(optimized_image, 0, 1)  # Clip to valid range

    # Display the optimized image
    st.image(optimized_image, caption="Optimized Image for Feature Visualization", use_container_width=True)

# Footer
#st.markdown("---")
#st.markdown("Created by [Your Name](https://github.com/yourusername)")