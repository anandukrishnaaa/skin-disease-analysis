import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as preprocess_input_mobilenetv2,
)
from tensorflow.keras.applications.resnet import (
    preprocess_input as preprocess_input_resnet50,
)
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from _config import MODEL_DIR, TRAIN_DIR, TEST_DIR, RESULT_DIR, TEST_IMG_PATH, logger


class SkinDiseaseModel:
    def __init__(self, model_type="mobilenetv2"):
        self.model_type = model_type.lower()
        self.model_weights_file = (
            MODEL_DIR
            / f"{self.model_type}_skin_disease_classification_model_weights.h5"
        )

        if self.model_type == "mobilenetv2":
            preprocess_input = preprocess_input_mobilenetv2
        elif self.model_type == "resnet50":
            preprocess_input = preprocess_input_resnet50
        else:
            raise ValueError(
                "Invalid model type. Supported types: resnet50, mobilenetv2"
            )
        self.preprocess_function = preprocess_input
        self.num_classes = self.get_num_classes(data_path=TRAIN_DIR)
        self.le = LabelEncoder()
        self.label_encoder = None  # Initialize label_encoder attribute

        # Create LabelEncoder during initialization
        logger.info("Creating LabelEncoder...")
        self.create_label_encoder(data_path=TRAIN_DIR)

    def create_label_encoder(self, data_path):
        # Fetch directory names and use them as class labels
        class_labels = [dir.name for dir in Path(data_path).iterdir() if dir.is_dir()]

        # Fit LabelEncoder on the class labels
        self.le.fit(class_labels)
        self.label_encoder = self.le

    def get_num_classes(self, data_path):
        # Fetch directory names and use them as class labels
        class_labels = [dir.name for dir in Path(data_path).iterdir() if dir.is_dir()]

        return len(class_labels)

    def check_directory_balance(self, train_dir, test_dir):
        train_classes = [dir.name for dir in Path(train_dir).iterdir() if dir.is_dir()]
        test_classes = [dir.name for dir in Path(test_dir).iterdir() if dir.is_dir()]

        is_balanced = len(train_classes) == len(test_classes)
        balance_message = (
            f"Train directory has {len(train_classes)} classes, "
            f"Test directory has {len(test_classes)} classes. "
            "Directories are balanced."
            if is_balanced
            else "Directories are not balanced."
        )

        return is_balanced, balance_message

    def create_base_model(self):
        if self.model_type == "mobilenetv2":
            base_model = MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
        elif self.model_type == "resnet50":
            base_model = ResNet50(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
        else:
            raise ValueError("Invalid model type. Supported types: mobilenetv2")

        return base_model

    def create_skin_disease_model(self, base_model):
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        )(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def train_model(self, data_path, batch_size=2, epochs=12, override=False):
        # Check directory balance
        is_balanced, balance_message = self.check_directory_balance(TRAIN_DIR, TEST_DIR)
        logger.info(balance_message)

        if not is_balanced:
            return "Directory balance check failed. Training aborted."
        if self.model_weights_file.exists() and override is not True:
            logger.info(
                f"Model weights file '{self.model_weights_file}' already exists. Skipping training."
            )
            return f"Model weights file '{self.model_weights_file}' already exists. Skipping training."

        logger.info(f"Training skin disease classification model {self.model_type} ...")

        base_model = self.create_base_model()
        model = self.create_skin_disease_model(base_model)

        # Load and preprocess training data using a data generator
        datagen = ImageDataGenerator(
            rescale=1.0 / 255, preprocessing_function=self.preprocess_function
        )
        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
        )

        # Log and save the time taken for training
        start_time = time.time()
        try:
            # Train the skin disease classification model using the data generator
            history = model.fit(train_generator, epochs=epochs, workers=2)

            # Save the entire model (architecture, weights, optimizer state)
            save_model(model, self.model_weights_file)

            end_time = time.time()
            time_taken = end_time - start_time

            logger.info(
                f"Training completed. Model saved to '{self.model_weights_file}'."
            )
            logger.info(f"Time taken for training: {time_taken:.2f} seconds")

            # Get the training accuracies from the history object
            train_acc = history.history["accuracy"]

            # Plot the training accuracies
            plt.plot(
                range(1, epochs + 1),
                train_acc,
                label="Training accuracy",
                marker="o",
            )
            plt.title("Training Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(RESULT_DIR / f"training_accuracy_{self.model_type}.png")
            return f"Training completed successfully! Model saved to '{self.model_weights_file}'.Time taken for training: {time_taken:.2f} seconds."

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return f"Error during training: {str(e)}"

        except KeyboardInterrupt as e:
            logger.error(f"Error during training: {str(e)}")
            print("Training manually aborted!")

        finally:
            # Evaluate the model on the test set
            evaluate_result = self.evaluate_model(TEST_DIR)

    def evaluate_model(self, test_path, batch_size=2, num_predictions=3):
        try:
            base_model = self.create_base_model()
            model = self.create_skin_disease_model(base_model)

            # Load weights into the model
            model.load_weights(self.model_weights_file)

            # Test the model on the test set using a data generator
            datagen = ImageDataGenerator(
                rescale=1.0 / 255, preprocessing_function=self.preprocess_function
            )
            test_generator = datagen.flow_from_directory(
                test_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )

            # Log and save the time taken for evaluation
            start_time = time.time()

            # Get the true labels and predicted labels
            y_true = test_generator.classes
            y_pred = model.predict(test_generator, steps=len(test_generator))
            y_pred_classes = np.argsort(y_pred, axis=1)[:, ::-1][:, :num_predictions]
            top_confidences = np.max(y_pred, axis=1)

            # # Print unique class labels obtained from LabelEncoder
            unique_labels = self.le.classes_
            # print(f"Unique class labels: {unique_labels}")

            # Map predicted indices to class names using the LabelEncoder
            predicted_classes = self.label_encoder.classes_[y_pred_classes[:, 0]]

            # Convert variables to strings for serialization
            y_true_str = y_true.astype(str).tolist()
            top_confidences_str = top_confidences.astype(str).tolist()

            # Prepare the result in the desired format
            result = {self.model_type: []}

            for i in range(len(y_true_str)):
                predictions_info = {}
                for j in range(num_predictions):
                    prediction_key = f"prediction_{j + 1}"
                    predictions_info[prediction_key] = {
                        "class": str(predicted_classes[i]),
                        "confidence": f"{float(top_confidences_str[i]) * 100:.2f}%",
                    }
                result[self.model_type].append(predictions_info)

            # Print the result in JSON format to the console
            json_output = json.dumps(result, indent=2)

            # Plot the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred_classes[:, 0])
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(RESULT_DIR / f"confusion_matrix_{self.model_type}.png")

            end_time = time.time()
            time_taken = end_time - start_time

            logger.info(
                f"Evaluation completed. Confusion matrix saved to 'confusion_matrix_{self.model_type}.png'."
            )
            logger.info(f"Time taken for evaluation: {time_taken:.2f} seconds")
            return json_output, time_taken

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return f"Error during evaluation: {str(e)}"

    def predict_class(self, img_path, num_predictions=1):
        # Log and save the time taken for prediction
        start_time = time.time()
        try:
            base_model = self.create_base_model()
            model = self.create_skin_disease_model(base_model)

            # Load weights into the model
            model.load_weights(self.model_weights_file)

            # Example of image preprocessing and classification
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = self.preprocess_function(np.array([img]))

            # Make predictions using the trained model on the image
            predictions = model.predict(img)

            # Get the top predicted classes and their confidence levels
            top_predictions = np.argsort(predictions[0])[::-1][:num_predictions]
            top_confidences = predictions[0][top_predictions]

            # Map predicted indices to class names using the LabelEncoder
            predicted_classes = self.label_encoder.classes_[top_predictions]

            # Convert variables to strings for serialization
            top_confidences_str = [f"{conf * 100:.2f}%" for conf in top_confidences]

            # Prepare the result in JSON format
            result = {self.model_type: []}

            for i in range(num_predictions):
                prediction_info = {
                    f"prediction_{i + 1}": {
                        "class": str(predicted_classes[i]),
                        "confidence": f"{top_confidences_str[i]}",
                    }
                }
                result[self.model_type].append(prediction_info)

            end_time = time.time()
            time_taken = end_time - start_time

            # Print the result in JSON format to the console
            json_output = json.dumps(result, indent=2)
            return json_output, time_taken

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return f"Error during prediction: {str(e)}"
