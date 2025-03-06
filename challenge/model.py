import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Configure Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DelayModel:
    """
    A model to predict flight delays using an XGBoost classifier.

    This model preprocesses flight data, trains an XGBoost classifier with
    class balancing, and predicts whether flights will be delayed.

    Attributes:
        JSON_MODEL_PATH (str): File path for saving/loading the booster in JSON format.
        _model (xgb.XGBClassifier): Underlying XGBoost classifier instance.
        _feature_cols (List[str]): List of feature names used by the model.
        _top_features (List[str]): A set of columns considered most relevant based on feature importance analysis.
    """

    JSON_MODEL_PATH = "model.json"  # Booster en formato JSON (portátil)

    def __init__(self):
        """
        Initialize the model with default attributes.
        """
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_cols: List[str] = []

        # Top 10 features based on previous feature-importance analysis (example)
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
        logger.info("DelayModel initialized with top 10 features.")

    # -----------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------
    def preprocess(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        If `target_column` is provided, returns (features, target).
        Otherwise, returns only features.

        Args:
            data (pd.DataFrame): Raw data to preprocess, containing
                'OPERA', 'TIPOVUELO', 'MES', 'Fecha-O', 'Fecha-I', etc.
            target_column (str, optional): Name of the target column in `data`.
                If provided, the returned tuple includes the target.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
                - If `target_column` is given: (features, target)
                - Else: features
        """
        logger.info("Starting data preprocessing.")
        try:
            df_data = data.copy()

            # One-hot encoding for categorical variables
            features = pd.concat(
                [
                    pd.get_dummies(df_data["OPERA"], prefix="OPERA"),
                    pd.get_dummies(df_data["TIPOVUELO"], prefix="TIPOVUELO"),
                    pd.get_dummies(df_data["MES"], prefix="MES"),
                ],
                axis=1,
            )

            logger.info(f"Created {features.shape[1]} features via one-hot encoding.")

            # Ensure all top features exist (create columns of 0 if missing)
            for feature in self._top_features:
                if feature not in features.columns:
                    features[feature] = 0
                    logger.debug(f"Added missing feature column: {feature}")

            # Select only the top features for model input
            features = features[self._top_features]
            logger.info(f"Selected {features.shape[1]} top features for the model.")

            if target_column is not None:
                # Example: create a binary delay variable using 'Fecha-O' and 'Fecha-I'
                if "Fecha-O" in df_data.columns and "Fecha-I" in df_data.columns:
                    df_data["min_diff"] = df_data.apply(self._compute_time_diff, axis=1)

                    # If flight starts 15 min or more after scheduled time => delayed
                    threshold_in_minutes = 15
                    df_data["delay"] = np.where(
                        df_data["min_diff"] > threshold_in_minutes, 1, 0
                    )
                else:
                    logger.warning(
                        "Could not find 'Fecha-O' or 'Fecha-I' columns. "
                        "No 'delay' variable generated from time difference."
                    )

                # Return features + target if present
                if target_column in df_data.columns:
                    target = pd.DataFrame(df_data[target_column]).copy()
                    logger.info(f"Returning features + target '{target_column}'.")
                    return features, target
                else:
                    raise ValueError(
                        f"Target column '{target_column}' not found in data."
                    )

            # Otherwise, return just the features
            logger.info("Returning features only (no target).")
            return features

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    @staticmethod
    def _compute_time_diff(row: pd.Series) -> float:
        """
        Compute the time difference in minutes between 'Fecha-O' and 'Fecha-I'.

        Args:
            row (pd.Series): A row with 'Fecha-O' and 'Fecha-I'.

        Returns:
            float: The difference in minutes between Fecha-O and Fecha-I.
        """
        try:
            fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
            fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
            return (fecha_o - fecha_i).total_seconds() / 60.0
        except Exception as e:
            logger.error(f"Failed to compute time diff for row: {e}")
            return 0.0  # Or raise, depending on your preference

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit (train) an XGBoost model with preprocessed data.

        Args:
            features (pd.DataFrame): Input features (already one-hot encoded).
            target (pd.DataFrame): Target values for each sample.
        """
        logger.info("Starting model fitting.")
        try:
            # Compute class imbalance ratio
            n_y0 = sum(target.iloc[:, 0] == 0)
            n_y1 = sum(target.iloc[:, 0] == 1)

            if n_y1 == 0:
                logger.warning("No delayed flights (class 1) in training data.")
                scale = 1.0
            else:
                scale = n_y0 / n_y1
                logger.info(f"Class imbalance ratio ~ {scale:.2f}")

            # Preliminary classifier for evaluation
            temp_model = xgb.XGBClassifier(
                random_state=1, learning_rate=0.01, scale_pos_weight=scale
            )

            # Train-test split for a quick local validation
            X_train, X_test, y_train, y_test = train_test_split(
                features, target.values.ravel(), test_size=0.2, random_state=42
            )

            logger.info(
                f"Split data: {X_train.shape[0]} train samples, "
                f"{X_test.shape[0]} test samples."
            )

            temp_model.fit(X_train, y_train)
            y_pred = temp_model.predict(X_test)

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")

            # Classification report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            logger.info("Classification Report:")
            logger.info(
                f"  Class 0 -> P: {report_dict['0']['precision']:.2f}, "
                f"R: {report_dict['0']['recall']:.2f}, "
                f"F1: {report_dict['0']['f1-score']:.2f}"
            )
            logger.info(
                f"  Class 1 -> P: {report_dict['1']['precision']:.2f}, "
                f"R: {report_dict['1']['recall']:.2f}, "
                f"F1: {report_dict['1']['f1-score']:.2f}"
            )
            logger.info(f"  Accuracy: {report_dict['accuracy']:.2f}")

            # Train final model on all data
            final_model = xgb.XGBClassifier(
                random_state=1, learning_rate=0.01, scale_pos_weight=scale
            )
            final_model.fit(features, target.values.ravel())
            self._model = final_model
            self._feature_cols = list(features.columns)

            logger.info("Model successfully trained on the complete dataset.")
            self.save_booster_as_json(self.JSON_MODEL_PATH)
            logger.info(
                f"Model training completed and saved at '{self.JSON_MODEL_PATH}'."
            )

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed features.

        Returns:
            List[int]: Predicted binary classes (1=delayed, 0=on time).

        Raises:
            ValueError: If the model cannot be loaded or is not available.
        """
        logger.info("Starting prediction process.")
        try:
            # Ensure model is loaded
            if self._model is None:
                logger.warning("Model is not loaded; attempting to load from disk.")
                try:
                    self.load_booster_from_json(self.JSON_MODEL_PATH)

                    # Verify model was successfully loaded
                    if self._model is None:
                        raise ValueError(
                            "Failed to load model from disk. Please ensure the model file exists and is valid."
                        )
                except FileNotFoundError as fnf:
                    logger.error(f"Model file not found: {fnf}")
                    raise ValueError(
                        f"Model file not found at '{self.JSON_MODEL_PATH}'. Please train the model first."
                    ) from fnf
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise ValueError(f"Error loading model: {e}") from e

            # Align columns - verify feature columns exist
            if not self._feature_cols:
                raise ValueError(
                    "Feature column list is empty. Model metadata may be corrupted."
                )

            # Align columns
            aligned_features = features.reindex(
                columns=self._feature_cols, fill_value=0
            )

            # Make predictions
            preds = self._model.predict(aligned_features)
            return preds.tolist()

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    # -----------------------------------------------------------------
    # JSON Booster Save/Load for Portability
    # -----------------------------------------------------------------
    def save_booster_as_json(self, filepath: str = JSON_MODEL_PATH) -> None:
        """
        Save the booster from the trained XGBoost model in JSON format (portable).

        Args:
            filepath (str): Path to JSON file. Default is `JSON_MODEL_PATH`.
        """
        if not self._model:
            logger.warning("No model to save as JSON.")
            return

        booster = self._model.get_booster()
        booster.save_model(filepath)
        logger.info(f"Booster saved in JSON format at '{filepath}'.")

        # Save additional metadata (feature_cols, etc.) so you can reconstruct the classifier
        metadata = {"features": self._feature_cols, "params": self._model.get_params()}
        joblib.dump(metadata, filepath + ".metadata")
        logger.info(f"Metadata saved at '{filepath}.metadata'.")

    def load_booster_from_json(self, filepath: str = JSON_MODEL_PATH) -> None:
        """
        Load a booster from JSON and reconstruct an XGBClassifier.

        Args:
            filepath (str): Path to the booster JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file '{filepath}' does not exist.")

        booster = xgb.Booster()
        booster.load_model(filepath)
        logger.info(f"Booster loaded from '{filepath}'.")

        # Rebuild an XGBClassifier
        xgb_clf = xgb.XGBClassifier()
        xgb_clf._Booster = booster  # Attach the booster
        logger.info("Reconstructed XGBClassifier from the loaded booster.")

        # Load metadata to restore features and model params
        if os.path.exists(filepath + ".metadata"):
            metadata = joblib.load(filepath + ".metadata")
            xgb_clf.set_params(**metadata["params"])
            self._feature_cols = metadata["features"]
            logger.info(
                f"Restored classifier parameters and {len(self._feature_cols)} features."
            )
        else:
            logger.warning(
                "No metadata file found. Feature columns or params may be missing."
            )

        self._model = xgb_clf
        logger.info("Model reconstruction from JSON complete.")
