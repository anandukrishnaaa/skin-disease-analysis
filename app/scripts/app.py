import streamlit as st
from _ml import SkinDiseaseModel
from _config import MODEL_DIR, TRAIN_DIR, TEST_DIR, TEST_IMG_PATH, RESULT_DIR, LOG_DIR
import datetime
from PIL import Image
from pathlib import Path
import pandas as pd
import json
import os

# Set theme
st.set_page_config(
    page_title="Skin disease prediction application",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Made to detect skin anomalies and diseases from image using Machine learning.",
    },
)

model_types = ["MobileNetV2", "ResNet50"]

st.sidebar.subheader("Models available")
# Add a dropdown for selecting the model_type
selected_model_type = st.sidebar.selectbox(
    "Select Model Type", model_types, key="model_type"
)

if selected_model_type == "MobileNetV2":
    model_type = "mobilenetv2"
elif selected_model_type == "ResNet50":
    model_type = "resnet50"

# Create an instance of the SkinDiseaseModel
model_instance = SkinDiseaseModel(model_type=model_type)


def directory_status():
    directory_data = {}
    st.sidebar.subheader("Directory and file health check")

    # Data for the table
    directory_data = {
        "Model Directory": MODEL_DIR,
        "Model File": model_instance.model_weights_file,
        "Training Directory": TRAIN_DIR,
        "Test Directory": TEST_DIR,
        "Result Directory": RESULT_DIR,
        "Image upload directory": TEST_IMG_PATH,
        "Log directory": LOG_DIR,
    }

    # Check if the directories exist
    status_data = {
        key: ("Available ✅" if Path(value).exists() else "Not Available ❌")
        for key, value in directory_data.items()
    }

    return status_data


def directory_info(data_path):
    st.sidebar.subheader("Dataset details")
    directory_count = 0
    directory_info = {}

    for directory in Path(data_path).iterdir():
        if directory.is_dir():
            directory_name = directory.name
            num_images = len(list(directory.glob("*")))  # Count files in the directory
            directory_info[directory_name] = num_images

    directory_count = sum(
        os.path.isdir(os.path.join(data_path, item)) for item in os.listdir(data_path)
    )

    return directory_info, directory_count


def predict_tab():
    col1, col2 = st.columns(2)

    with col1:
        st.title("Predict by uploading an image")
        st.write(
            "Upload an image to run through the machine model for detecting skin anomalies."
        )
        json_toggle = st.toggle(
            "Display prediction result JSON",
            help="Displays the raw JSON output from prediction function.",
        )
    with col2:
        # File upload
        uploaded_file = st.file_uploader(
            "Perform prediction by uploading an image.",
            type="jpg",
            help="Upload a JPEG file to analyse the skin anomaly present. Files up tp 200 mb can be uploaded.",
        )
        # Number of predictions text box
        num_predictions = st.slider(
            "Number of results to return",
            value=3,
            key="num_predictions",
            min_value=1,
            step=1,
            max_value=10,
        )
        predict_button = st.button("Predict")
    with col1:
        # File upload handler
        # Display the uploaded image
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(
                uploaded_image,
                caption=f"{uploaded_file.name}",
                use_column_width="auto",
                width=600,
            )

    with col1:
        with st.container():
            if predict_button and uploaded_file is not None:
                timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M")
                original_filename = uploaded_file.name
                image_path = f"{TEST_IMG_PATH}/{original_filename[:-4]}_{timestamp}.jpg"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("Classifying skin anomalay..."):
                    predict_result, time_taken = model_instance.predict_class(
                        image_path, num_predictions=num_predictions
                    )

                if not predict_result.startswith("Error"):
                    if json_toggle:
                        st.json(predict_result)
                    with col2:
                        predict_result = json.loads(predict_result)
                        for prediction in predict_result[model_type]:
                            for key, value in prediction.items():
                                predict_result_id = key.rsplit("_", 1)[1]
                                st.metric(
                                    label=f"Prediction no. {predict_result_id}",
                                    value=f"{value['class']}",
                                    delta=f"{value['confidence']}",
                                )
                    st.toast(f"Prediction complete in {time_taken :.2f} seconds 🎉")

                else:
                    st.error(predict_result)
            elif predict_button and uploaded_file is None:
                st.toast("Upload an image first.")


def train_tab():
    st.title("Training the model")
    st.write(
        "Train the model again after replacing the dataset or after changing the underlying code."
    )

    col1, col2 = st.columns(2)

    with col1:  # Train
        with st.expander("Advanced Settings", expanded=False):
            # Advanced training settings
            st.warning(
                "Only change this if you're absolutely sure about what you're doing or it may break the system.",
                icon="⚠️",
            )
            num_batch = st.slider(
                "Batch size",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="Batch size refers to the number of training examples utilized in one iteration. By default, a batch size of 2 means that the model will update its weights after processing 2 images. This is a common practice to use small batches (mini-batches) to improve the efficiency of training. Smaller batch sizes consume less memory, which can be beneficial if you have limited GPU resources. Larger batch sizes can benefit from parallelization during training but may require more GPU memory.",
            )
            st.caption("Default batch size: 2")
            num_epoch = st.slider(
                "Number of epochs",
                min_value=1,
                max_value=20,
                value=12,
                step=1,
                help="An epoch is one complete pass through the entire training dataset. By default 12 epochs means that the model will go through the entire training dataset 12 times during the training process. Training for too few epochs may result in underfitting, where the model has not learned the underlying patterns in the data. Training for too many epochs may lead to overfitting, where the model learns noise in the training data, making it perform poorly on new, unseen data.",
            )
            st.caption("Default number of epochs: 12")
            # Override toggle
            override_toggle = st.toggle(
                label="Override check for existing model file",
                key="override",
                value=False,
                help="By default, if a model file exists than the train function is exited. Turning this toggle on would override the default and result in initiating the training process.",
            )
            st.caption("Default: False")

        if override_toggle:
            override = True
            st.toast(
                "This overrides the default check for existing model files, generating new model files will take some time."
            )
        else:
            override = False

        if st.button("Train Model"):
            with st.spinner("Model training in progress..."):
                model_train = model_instance.train_model(
                    TRAIN_DIR, override=override, batch_size=num_batch, epochs=num_epoch
                )
            if model_train is not None and not model_train.startswith("Error"):
                st.success(model_train)
                st.toast("Model trained successfully 🎉")
            elif model_train is not None:
                st.error(model_train)

    with col2:
        # Display the most recent training accuracy image
        train_accuracy_image_path = RESULT_DIR / f"training_accuracy_{model_type}.png"
        if train_accuracy_image_path.exists():
            train_accuracy_image = Image.open(train_accuracy_image_path.as_posix())
            st.image(
                train_accuracy_image,
                caption="Most Recent Training Accuracy",
                width=600,
                use_column_width="auto",
            )


def evaluate_tab():
    st.title("Evaluating the model")
    st.write(
        "Evaluating the model by testing it on test data for gauging model performance."
    )

    col1, col2 = st.columns(2)

    with col1:
        # Evaluate
        if st.button("Evaluate Model"):
            with st.spinner("Model evaluation in progress..."):
                evaluate_model, time_taken = model_instance.evaluate_model(TEST_DIR)
            if evaluate_model is not None and not evaluate_model.startswith("Error"):
                st.json(
                    evaluate_model,
                )
                st.toast(f"Evaluation complete in {time_taken :.2f} seconds 🎉")
            elif evaluate_model is not None:
                st.error(evaluate_model)

    with col2:
        # Display the confusion matrix image
        confusion_matrix_image_path = RESULT_DIR / f"confusion_matrix_{model_type}.png"
        if confusion_matrix_image_path.exists():
            confusion_matrix_image_path.as_posix()
            confusion_matrix_image = Image.open(confusion_matrix_image_path.as_posix())

            st.image(
                confusion_matrix_image,
                caption="Evaluate Function Confusion Matrix",
                width=600,
                use_column_width="auto",
            )


def main():
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Skin disease prediction 🔬")
    # Display directory status table in the sidebar
    status_data = directory_status()
    # Display the table without column names
    for key, value in status_data.items():
        st.sidebar.caption(f"**{key}**: {value}")

    dataset_info, directory_count = directory_info(data_path=TRAIN_DIR)
    st.sidebar.write(f"Total number of classes available: **{directory_count}**")
    # Display the table without column names
    for key, value in dataset_info.items():
        st.sidebar.caption(f"**{key}**: {value}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Predict", "Train", "Evaluate"])

    # Main content based on selected tab
    with tab1:
        predict_tab()
    with tab2:
        train_tab()
    with tab3:
        evaluate_tab()


if __name__ == "__main__":
    main()
