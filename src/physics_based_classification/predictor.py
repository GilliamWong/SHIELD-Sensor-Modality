import numpy as np
import joblib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics_based_classification.feature_extractor import FeatureExtractor


class SensorPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models')
        model_dir = Path(model_path)
        self.clf = joblib.load(model_dir / 'rf_classifier.joblib')
        self.scaler = joblib.load(model_dir / 'scaler.joblib')
        self.le = joblib.load(model_dir / 'label_encoder.joblib')
        self.feature_cols = joblib.load(model_dir / 'feature_columns.joblib')
        self.extractor = None

    def _get_extractor(self, fs: float) -> FeatureExtractor:
        if self.extractor is None or self.extractor.fs != fs:
            self.extractor = FeatureExtractor(fs=fs)
        return self.extractor

    def predict(self, signal: np.ndarray, fs: float = 100.0) -> dict:
        signal = np.asarray(signal).flatten()
        if len(signal) < 200:
            raise ValueError("Signal too short. Need at least 200 samples.")

        extractor = self._get_extractor(fs)
        df = extractor.process_signal(signal, window_size_sec=2.0, step_size_sec=1.0)

        if df.empty:
            raise ValueError("Could not extract features. Signal may be too short.")

        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        preds = self.clf.predict(X_scaled)
        probs = self.clf.predict_proba(X_scaled)

        pred_labels = self.le.inverse_transform(preds)
        votes = {}
        for label in pred_labels:
            votes[label] = votes.get(label, 0) + 1

        most_common = max(votes, key=votes.get)
        confidence = votes[most_common] / len(pred_labels)

        avg_probs = probs.mean(axis=0)
        prob_dict = {self.le.classes_[i]: float(avg_probs[i]) for i in range(len(self.le.classes_))}

        return {
            'modality': most_common,
            'confidence': confidence,
            'probabilities': prob_dict,
            'n_windows': len(df)
        }

    def predict_batch(self, signals: list, fs: float = 100.0) -> list:
        return [self.predict(sig, fs) for sig in signals]


if __name__ == '__main__':
    from SensorDataLoader import SensorDataLoader

    loader = SensorDataLoader(seed=42)

    print("Testing SensorPredictor on synthetic data...")
    print("-" * 50)

    try:
        predictor = SensorPredictor()

        test_cases = [
            ('white', 'Expected: accelerometer-like'),
            ('pink', 'Expected: gyroscope/magnetometer-like'),
            ('brown', 'Expected: temperature/pressure-like')
        ]

        for noise_type, expected in test_cases:
            signal = loader.generate_synthetic_data(5000, noise_type=noise_type).flatten()
            result = predictor.predict(signal, fs=100.0)

            print(f"\n{noise_type.upper()} NOISE ({expected}):")
            print(f"  Predicted: {result['modality']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Windows analyzed: {result['n_windows']}")

    except FileNotFoundError:
        print("Model files not found. Run feature_reference.ipynb first to train and save the model.")
        sys.exit(1)
