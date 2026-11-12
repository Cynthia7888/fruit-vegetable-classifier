"""Streamlit image classifier for the trained fruit/vegetable models."""

from pathlib import Path

import streamlit as st
from PIL import Image

from fruit_vegetable_models import MODEL_NAMES, load_classifier, predict_image


MODEL_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource
def get_classifier(model_name: str, checkpoint_path: str):
    return load_classifier(model_name, checkpoint_path)


st.set_page_config(page_title="Fruit & Vegetable Classifier", page_icon="🍎")
st.title("Fruit & Vegetable Classifier")
st.write("Upload an image to classify it with a model trained in the Colab notebook.")

available_models = [
    name for name in MODEL_NAMES if (MODEL_DIR / f"{name}_best.pth").is_file()
]
if not available_models:
    st.error(
        f"No model checkpoints found in `{MODEL_DIR}`. "
        "Copy one or more `*_best.pth` files from the notebook's models folder here."
    )
    st.stop()

selected_model = st.selectbox("Model", available_models)
uploaded_file = st.file_uploader(
    "Choose an image", type=("jpg", "jpeg", "png", "webp", "bmp")
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    checkpoint_path = str(MODEL_DIR / f"{selected_model}_best.pth")

    try:
        model, classes, device = get_classifier(selected_model, checkpoint_path)
        predictions = predict_image(model, image, classes, device, top_k=5)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load or run the selected model: {exc}")
    else:
        st.subheader(f"Top prediction: {predictions[0]['class']}")
        st.caption(f"Running on {device}")
        st.bar_chart(
            {
                prediction["class"]: prediction["probability"]
                for prediction in predictions
            }
        )
