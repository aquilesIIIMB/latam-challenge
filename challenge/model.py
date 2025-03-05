import logging
from datetime import datetime
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DelayModel:
    """
    A model to predict flight delays using XGBoost classifier.

    This model preprocesses flight data, trains an XGBoost classifier with
    class balancing, and predicts whether flights will be delayed.
    """

    MODEL_PATH = "model.joblib"

    def __init__(self):
        """Initialize the model with default attributes."""
        self._model = None  # Model will be saved in this attribute
        self._feature_cols = []
        # Top 10 features based on feature importance analysis
        self._top_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        logger.info("DelayModel initialized with top 10 features")

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data to preprocess.
            target_column (str, optional): If set, the target is returned.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
                Features and target (if target_column is provided),
                or just features.
        """
        logger.info("Starting data preprocessing")

        try:

            def get_min_diff(data):
                fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
                fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
                min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
                return min_diff

            df_data = data.copy()

            df_data["min_diff"] = df_data.apply(get_min_diff, axis=1)

            threshold_in_minutes = 15
            df_data["delay"] = np.where(
                df_data["min_diff"] > threshold_in_minutes, 1, 0
            )

            # Create features using one-hot encoding for categorical variables
            features = pd.concat(
                [
                    pd.get_dummies(df_data["OPERA"], prefix="OPERA"),
                    pd.get_dummies(df_data["TIPOVUELO"], prefix="TIPOVUELO"),
                    pd.get_dummies(df_data["MES"], prefix="MES"),
                ],
                axis=1,
            )

            logger.info(
                f"Created {features.shape[1]} features through one-hot encoding"
            )

            # Ensure all top features are present (fill with zeros if missing)
            for feature in self._top_features:
                if feature not in features.columns:
                    features[feature] = 0
                    logger.info(f"Added missing feature column: {feature}")

            # Select only the top features for the model
            features = features[self._top_features]
            logger.info(f"Selected {features.shape[1]} top features for model")

            # If target column is provided, return features and target
            if target_column is not None:
                if target_column in df_data.columns:
                    # Prepare target as a DataFrame with delay column
                    target = pd.DataFrame(df_data[target_column]).copy()
                    logger.info(f"Target column '{target_column}' prepared")
                    return features, target
                else:
                    logger.error(f"Target column '{target_column}' not found in data")
                    raise ValueError(
                        f"Target column '{target_column}' not found in data"
                    )

            return features

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): Preprocessed features.
            target (pd.DataFrame): Target values (delay column).
        """
        logger.info("Starting model fitting")

        try:
            # Calculate class balance for weighting
            n_y0 = len(target[target.iloc[:, 0] == 0])  # Number of non-delayed flights
            n_y1 = len(target[target.iloc[:, 0] == 1])  # Number of delayed flights

            if n_y1 == 0:
                logger.warning("No delayed flights in training data")
                scale = 1.0
            else:
                scale = n_y0 / n_y1  # Weight for positive class
                logger.info(f"Class imbalance ratio: {scale:.2f} (non-delayed/delayed)")

            # Initialize XGBoost classifier with balanced class weights
            model_for_test = xgb.XGBClassifier(
                random_state=1, learning_rate=0.01, scale_pos_weight=scale
            )

            # Perform train-test split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                features, target.values.ravel(), test_size=0.2, random_state=42
            )

            logger.info(
                f"Training on {X_train.shape[0]} samples, evaluating on {X_test.shape[0]} samples"
            )

            # Train the model on the training split
            model_for_test.fit(X_train, y_train)

            # Evaluate the model on the test split
            y_pred = model_for_test.predict(X_test)

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")

            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info("Classification Report:")
            logger.info(
                f"Class 0 - Precision: {report['0']['precision']:.2f}, Recall: {report['0']['recall']:.2f}, F1: {report['0']['f1-score']:.2f}"
            )
            logger.info(
                f"Class 1 - Precision: {report['1']['precision']:.2f}, Recall: {report['1']['recall']:.2f}, F1: {report['1']['f1-score']:.2f}"
            )
            logger.info(f"Accuracy: {report['accuracy']:.2f}")

            # Now train the final model on all data
            logger.info("Training final model on complete dataset")
            trained_model = xgb.XGBClassifier(
                random_state=1, learning_rate=0.01, scale_pos_weight=scale
            )
            trained_model.fit(features, target.values.ravel())

            self._model = trained_model
            logger.info("Model successfully trained")

            # Store feature columns
            self._feature_cols = list(features.columns)

            # Save the trained model
            self.save_model(self.MODEL_PATH)
            logging.info("Model training completed and saved.")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed features.

        Returns:
            List[int]: Predicted targets (1 for delay, 0 for no delay).
        """
        logging.info("Starting prediction process")

        # Ensure model is trained or loaded
        if self._model is None:
            logging.warning("Model is not loaded. Attempting to load from disk...")
            try:
                self.load_model(self.MODEL_PATH)
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error(f"Model could not be loaded: {e}")
                raise RuntimeError(
                    "Model is not trained or cannot be loaded. Call `fit()` first."
                )

        try:
            # Ensure feature alignment
            features = features.reindex(columns=self._feature_cols)

            # Make predictions
            predictions = self._model.predict(features)
            return predictions.tolist()

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump({"model": self._model, "features": self._feature_cols}, filepath)
        logging.info(f"Model saved at {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to load the model.
        """
        model_data = joblib.load(filepath)
        self._model = model_data["model"]
        self._feature_cols = model_data["features"]
        logging.info(f"Model loaded from {filepath}")
