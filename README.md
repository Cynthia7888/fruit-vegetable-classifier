# Fruit & Vegetable Classifier

This project uses the Custom CNN, ResNet50, and VGG16 architectures. Train
models in Google Colab, then run this Streamlit app locally for image
classification. Local model training and a command-line classifier are not
needed for this workflow.

## Files

- `fruit_vegetable_classification_colab.ipynb` — trains models and evaluates
  their test-set performance in Google Colab.
- `streamlit_app.py` — local web app for image classification.
- `fruit_vegetable_models.py` — model definitions, checkpoint loading, and
  image preprocessing shared by the app.
- `requirements.txt` — dependencies for running the local Streamlit app.

## 1. Train in Google Colab

Open `fruit_vegetable_classification_colab.ipynb` in Colab, select a GPU
runtime if available, and run the cells in order. The notebook downloads the
public Fruit and Vegetable Image Recognition dataset using KaggleHub. It
expects `train/`, `validation/`, and `test/` folders, each with one subfolder
per class.

The notebook trains Custom CNN, ResNet50, and VGG16 by default. To train only
some architectures, edit `MODEL_NAMES` in the notebook. At the end, download
the generated `fruit_vegetable_models.zip`.

## 2. Install locally

Use the installed Python 3.15 runtime. In PowerShell, from this project
directory:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 5 -r requirements.txt
```

PyTorch is a large download and can take several minutes. The command above
uses longer network timeouts and retries. The requirements file is for local
Streamlit inference only. KaggleHub and scikit-learn are installed in the
Colab notebook and are not needed locally.

To activate the environment in a new PowerShell session, run
`.\.venv\Scripts\Activate.ps1` from the project directory.

## 3. Add the trained models

Extract `fruit_vegetable_models.zip` into this project folder. The app expects:

```text
models/
  custom_cnn_best.pth
  resnet50_best.pth
  vgg16_best.pth
```

Only checkpoints you trained are needed. Keep the `models` folder beside
`streamlit_app.py`.

## 4. Run Streamlit

Start the app from this project directory:

```powershell
streamlit run streamlit_app.py
```

Select an available model, upload a JPG, PNG, WEBP, or BMP image, and view the
top predictions. Input images are converted to RGB, resized to 224 × 224, and
normalized with ImageNet mean and standard deviation.
