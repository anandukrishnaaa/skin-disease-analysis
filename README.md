# Skin Disease Prediction Application

This application is designed to detect skin anomalies and diseases from images using machine learning. It provides functionalities to predict, train, and evaluate a skin disease classification model.

## About

This application is created using streamlit, TensorFlow, and other Python libraries. For more details, refer to the code in the repository.

## Usage

1. **Predict Tab:** Upload an image to run through the machine learning model for detecting skin anomalies. You can choose the model type (MobileNetV2 or ResNet50) and customize the number of prediction results.

2. **Train Tab:** Train the model again after replacing the dataset or changing the underlying code. Advanced settings are available for batch size, number of epochs, and an option to override the check for existing model files.

3. **Evaluate Tab:** Evaluate the model by testing it on a separate test dataset. It provides insights such as the confusion matrix and evaluation results.

## Directory and File Health Check

The application performs a health check on the following directories:

- Model Directory
- Training Directory
- Test Directory
- Result Directory
- Image Upload Directory
- Log Directory

The status of each directory is displayed in the sidebar.

## Models Available

Choose between `MobileNetV2` and `ResNet50` for skin disease prediction.

# Overview of Functions

## Main Application (app.py)

- `main()`: Main function coordinating the streamlit application.
  - Calls `directory_status()`, `directory_info()`, and displays health check and dataset info in the sidebar.
  - Implements tabs for "Predict," "Train," and "Evaluate" with corresponding functionalities.

## Skin Disease Model (_ml.py)

- `SkinDiseaseModel` Class:
  - `__init__(self, model_type)`: Initializes the model, sets model type, and creates a LabelEncoder.
  - `create_label_encoder(self, data_path)`: Creates a LabelEncoder for class labels.
  - `get_num_classes(self, data_path)`: Returns the number of classes in the dataset.
  - `check_directory_balance(self, train_dir, test_dir)`: Checks if training and test directories are balanced.
  - `create_base_model(self)`: Creates the base model (MobileNetV2 or ResNet50).
  - `create_skin_disease_model(self, base_model)`: Adds custom layers to create the skin disease classification model.
  - `train_model(self, data_path, batch_size, epochs, override)`: Trains the model on the specified dataset.
  - `evaluate_model(self, test_path, batch_size, num_predictions)`: Evaluates the model on the test set.
  - `predict_class(self, img_path, num_predictions)`: Predicts skin disease class for a given image.

## Additional Modules (_config.py)

- `_config.py`:
  - Configuration file containing directory paths, logger setup, and other constants.

## Key Features
1. **Predict Tab:**
   - Allows users to upload an image for skin anomaly detection.
   - Provides prediction results with class labels and confidence scores.
   - Option to display raw JSON output.

2. **Train Tab:**
   - Enables training the machine learning model with custom datasets.
   - Supports advanced settings such as batch size and number of epochs.
   - Displays the most recent training accuracy graph.

3. **Evaluate Tab:**
   - Evaluates the model performance on a test dataset.
   - Presents evaluation results, including a confusion matrix.
   
4. **Directory and Dataset Health Check:**
   - Sidebar displays the health status of various directories (e.g., model, training, test).
   - Provides insights into the dataset, such as the number of classes and images.

5. **Logging and Configuration:**
   - Utilizes logging to capture and handle errors during training and evaluation.
   - Configuration file (_config.py) centralizes constants and directory paths.

## How to Run

1. In the command prompt, set `PIPENV_VENV_IN_PROJECT=1`
2. Run `pipenv install`
3. Download dataset from [here](https://drive.google.com/drive/folders/1AiDVpgy-o4ZLKXZ_yqnWWEHFYhCbrfP1), unzip and copy it over to dataset folder.
4. Run the application: `streamlit run app.py`

## Dependency graph
```python
opencv-python==4.8.1.78
├── numpy [required: >=1.21.2, installed: 1.26.2]
├── numpy [required: >=1.23.5, installed: 1.26.2]
├── numpy [required: >=1.17.0, installed: 1.26.2]
├── numpy [required: >=1.17.3, installed: 1.26.2]
└── numpy [required: >=1.19.3, installed: 1.26.2]
scikit-learn==1.3.2
├── joblib [required: >=1.1.1, installed: 1.3.2]
├── numpy [required: >=1.17.3,<2.0, installed: 1.26.2]
├── scipy [required: >=1.5.0, installed: 1.11.4]
│   └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.2]
└── threadpoolctl [required: >=2.0.0, installed: 3.2.0]
seaborn==0.13.0
├── matplotlib [required: >=3.3,!=3.6.1, installed: 3.8.2]
│   ├── contourpy [required: >=1.0.1, installed: 1.2.0]
│   │   └── numpy [required: >=1.20,<2.0, installed: 1.26.2]
│   ├── cycler [required: >=0.10, installed: 0.12.1]
│   ├── fonttools [required: >=4.22.0, installed: 4.46.0]
│   ├── kiwisolver [required: >=1.3.1, installed: 1.4.5]
│   ├── numpy [required: >=1.21,<2, installed: 1.26.2]
│   ├── packaging [required: >=20.0, installed: 23.2]
│   ├── pillow [required: >=8, installed: 10.1.0]
│   ├── pyparsing [required: >=2.3.1, installed: 3.1.1]
│   └── python-dateutil [required: >=2.7, installed: 2.8.2]
│       └── six [required: >=1.5, installed: 1.16.0]
├── numpy [required: >=1.20,!=1.24.0, installed: 1.26.2]
└── pandas [required: >=1.2, installed: 2.1.4]
    ├── numpy [required: >=1.23.2,<2, installed: 1.26.2]
    ├── python-dateutil [required: >=2.8.2, installed: 2.8.2]
    │   └── six [required: >=1.5, installed: 1.16.0]
    ├── pytz [required: >=2020.1, installed: 2023.3.post1]
    └── tzdata [required: >=2022.1, installed: 2023.3]
streamlit==1.29.0
├── altair [required: >=4.0,<6, installed: 5.2.0]
│   ├── jinja2 [required: Any, installed: 3.1.2]
│   │   └── MarkupSafe [required: >=2.0, installed: 2.1.3]
│   ├── jsonschema [required: >=3.0, installed: 4.20.0]
│   │   ├── attrs [required: >=22.2.0, installed: 23.1.0]
│   │   ├── jsonschema-specifications [required: >=2023.03.6, installed: 2023.11.2]
│   │   │   └── referencing [required: >=0.31.0, installed: 0.32.0]
│   │   │       ├── attrs [required: >=22.2.0, installed: 23.1.0]
│   │   │       └── rpds-py [required: >=0.7.0, installed: 0.14.1]
│   │   ├── referencing [required: >=0.28.4, installed: 0.32.0]
│   │   │   ├── attrs [required: >=22.2.0, installed: 23.1.0]
│   │   │   └── rpds-py [required: >=0.7.0, installed: 0.14.1]
│   │   └── rpds-py [required: >=0.7.1, installed: 0.14.1]
│   ├── numpy [required: Any, installed: 1.26.2]
│   ├── packaging [required: Any, installed: 23.2]
│   ├── pandas [required: >=0.25, installed: 2.1.4]
│   │   ├── numpy [required: >=1.23.2,<2, installed: 1.26.2]
│   │   ├── python-dateutil [required: >=2.8.2, installed: 2.8.2]
│   │   │   └── six [required: >=1.5, installed: 1.16.0]
│   │   ├── pytz [required: >=2020.1, installed: 2023.3.post1]
│   │   └── tzdata [required: >=2022.1, installed: 2023.3]
│   └── toolz [required: Any, installed: 0.12.0]
├── blinker [required: >=1.0.0,<2, installed: 1.7.0]
├── cachetools [required: >=4.0,<6, installed: 5.3.2]
├── click [required: >=7.0,<9, installed: 8.1.7]
│   └── colorama [required: Any, installed: 0.4.6]
├── gitpython [required: >=3.0.7,<4,!=3.1.19, installed: 3.1.40]
│   └── gitdb [required: >=4.0.1,<5, installed: 4.0.11]
│       └── smmap [required: >=3.0.1,<6, installed: 5.0.1]
├── importlib-metadata [required: >=1.4,<7, installed: 6.11.0]
│   └── zipp [required: >=0.5, installed: 3.17.0]
├── numpy [required: >=1.19.3,<2, installed: 1.26.2]
├── packaging [required: >=16.8,<24, installed: 23.2]
├── pandas [required: >=1.3.0,<3, installed: 2.1.4]
│   ├── numpy [required: >=1.23.2,<2, installed: 1.26.2]
│   ├── python-dateutil [required: >=2.8.2, installed: 2.8.2]
│   │   └── six [required: >=1.5, installed: 1.16.0]
│   ├── pytz [required: >=2020.1, installed: 2023.3.post1]
│   └── tzdata [required: >=2022.1, installed: 2023.3]
├── pillow [required: >=7.1.0,<11, installed: 10.1.0]
├── protobuf [required: >=3.20,<5, installed: 4.23.4]
├── pyarrow [required: >=6.0, installed: 14.0.1]
│   └── numpy [required: >=1.16.6, installed: 1.26.2]
├── pydeck [required: >=0.8.0b4,<1, installed: 0.8.1b0]
│   ├── jinja2 [required: >=2.10.1, installed: 3.1.2]
│   │   └── MarkupSafe [required: >=2.0, installed: 2.1.3]
│   └── numpy [required: >=1.16.4, installed: 1.26.2]
├── python-dateutil [required: >=2.7.3,<3, installed: 2.8.2]
│   └── six [required: >=1.5, installed: 1.16.0]
├── requests [required: >=2.27,<3, installed: 2.31.0]
│   ├── certifi [required: >=2017.4.17, installed: 2023.11.17]
│   ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
│   ├── idna [required: >=2.5,<4, installed: 3.6]
│   └── urllib3 [required: >=1.21.1,<3, installed: 2.1.0]
├── rich [required: >=10.14.0,<14, installed: 13.7.0]
│   ├── markdown-it-py [required: >=2.2.0, installed: 3.0.0]
│   │   └── mdurl [required: ~=0.1, installed: 0.1.2]
│   └── pygments [required: >=2.13.0,<3.0.0, installed: 2.17.2]
├── tenacity [required: >=8.1.0,<9, installed: 8.2.3]
├── toml [required: >=0.10.1,<2, installed: 0.10.2]
├── tornado [required: >=6.0.3,<7, installed: 6.4]
├── typing-extensions [required: >=4.3.0,<5, installed: 4.9.0]
├── tzlocal [required: >=1.1,<6, installed: 5.2]
│   └── tzdata [required: Any, installed: 2023.3]
├── validators [required: >=0.2,<1, installed: 0.22.0]
└── watchdog [required: >=2.1.5, installed: 3.0.0]
tensorflow==2.15.0
└── tensorflow-intel [required: ==2.15.0, installed: 2.15.0]
    ├── absl-py [required: >=1.0.0, installed: 2.0.0]
    ├── astunparse [required: >=1.6.0, installed: 1.6.3]
    │   ├── six [required: >=1.6.1,<2.0, installed: 1.16.0]
    │   └── wheel [required: >=0.23.0,<1.0, installed: 0.42.0]
    ├── flatbuffers [required: >=23.5.26, installed: 23.5.26]
    ├── gast [required: >=0.2.1,!=0.5.2,!=0.5.1,!=0.5.0, installed: 0.5.4]
    ├── google-pasta [required: >=0.1.1, installed: 0.2.0]
    │   └── six [required: Any, installed: 1.16.0]
    ├── grpcio [required: >=1.24.3,<2.0, installed: 1.60.0]
    ├── h5py [required: >=2.9.0, installed: 3.10.0]
    │   └── numpy [required: >=1.17.3, installed: 1.26.2]
    ├── keras [required: >=2.15.0,<2.16, installed: 2.15.0]
    ├── libclang [required: >=13.0.0, installed: 16.0.6]
    ├── ml-dtypes [required: ~=0.2.0, installed: 0.2.0]
    │   ├── numpy [required: >1.20, installed: 1.26.2]
    │   ├── numpy [required: >=1.23.3, installed: 1.26.2]
    │   └── numpy [required: >=1.21.2, installed: 1.26.2]
    ├── numpy [required: >=1.23.5,<2.0.0, installed: 1.26.2]
    ├── opt-einsum [required: >=2.3.2, installed: 3.3.0]
    │   └── numpy [required: >=1.7, installed: 1.26.2]
    ├── packaging [required: Any, installed: 23.2]
    ├── protobuf [required: >=3.20.3,<5.0.0dev,!=4.21.5,!=4.21.4,!=4.21.3,!=4.21.2,!=4.21.1,!=4.21.0, installed: 4.23.4]
    ├── setuptools [required: Any, installed: 69.0.2]
    ├── six [required: >=1.12.0, installed: 1.16.0]
    ├── tensorboard [required: >=2.15,<2.16, installed: 2.15.1]
    │   ├── absl-py [required: >=0.4, installed: 2.0.0]
    │   ├── google-auth [required: >=1.6.3,<3, installed: 2.25.2]
    │   │   ├── cachetools [required: >=2.0.0,<6.0, installed: 5.3.2]
    │   │   ├── pyasn1-modules [required: >=0.2.1, installed: 0.3.0]
    │   │   │   └── pyasn1 [required: >=0.4.6,<0.6.0, installed: 0.5.1]
    │   │   └── rsa [required: >=3.1.4,<5, installed: 4.9]
    │   │       └── pyasn1 [required: >=0.1.3, installed: 0.5.1]
    │   ├── google-auth-oauthlib [required: >=0.5,<2, installed: 1.2.0]
    │   │   ├── google-auth [required: >=2.15.0, installed: 2.25.2]
    │   │   │   ├── cachetools [required: >=2.0.0,<6.0, installed: 5.3.2]
    │   │   │   ├── pyasn1-modules [required: >=0.2.1, installed: 0.3.0]
    │   │   │   │   └── pyasn1 [required: >=0.4.6,<0.6.0, installed: 0.5.1]
    │   │   │   └── rsa [required: >=3.1.4,<5, installed: 4.9]
    │   │   │       └── pyasn1 [required: >=0.1.3, installed: 0.5.1]
    │   │   └── requests-oauthlib [required: >=0.7.0, installed: 1.3.1]
    │   │       ├── oauthlib [required: >=3.0.0, installed: 3.2.2]
    │   │       └── requests [required: >=2.0.0, installed: 2.31.0]
    │   │           ├── certifi [required: >=2017.4.17, installed: 2023.11.17]
    │   │           ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
    │   │           ├── idna [required: >=2.5,<4, installed: 3.6]
    │   │           └── urllib3 [required: >=1.21.1,<3, installed: 2.1.0]
    │   ├── grpcio [required: >=1.48.2, installed: 1.60.0]
    │   ├── markdown [required: >=2.6.8, installed: 3.5.1]
    │   ├── numpy [required: >=1.12.0, installed: 1.26.2]
    │   ├── protobuf [required: >=3.19.6,<4.24, installed: 4.23.4]
    │   ├── requests [required: >=2.21.0,<3, installed: 2.31.0]
    │   │   ├── certifi [required: >=2017.4.17, installed: 2023.11.17]
    │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
    │   │   ├── idna [required: >=2.5,<4, installed: 3.6]
    │   │   └── urllib3 [required: >=1.21.1,<3, installed: 2.1.0]
    │   ├── setuptools [required: >=41.0.0, installed: 69.0.2]
    │   ├── six [required: >1.9, installed: 1.16.0]
    │   ├── tensorboard-data-server [required: >=0.7.0,<0.8.0, installed: 0.7.2]
    │   └── werkzeug [required: >=1.0.1, installed: 3.0.1]
    │       └── MarkupSafe [required: >=2.1.1, installed: 2.1.3]
    ├── tensorflow-estimator [required: >=2.15.0,<2.16, installed: 2.15.0]
    ├── tensorflow-io-gcs-filesystem [required: >=0.23.1, installed: 0.31.0]
    ├── termcolor [required: >=1.1.0, installed: 2.4.0]
    ├── typing-extensions [required: >=3.6.6, installed: 4.9.0]
    └── wrapt [required: >=1.11.0,<1.15, installed: 1.14.1]
```