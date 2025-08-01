#!/usr/bin/env python3
"""
Voice Clone Detection Training with Real Dataset
Uses the downloaded "fake or real" dataset to train accurate models
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import pickle
import joblib
from datetime import datetime
import warnings
from tqdm import tqdm

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# HuggingFace datasets
from datasets import load_from_disk

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class RealDatasetProcessor:
    """Process the real voice dataset for training"""
    
    def __init__(self, dataset_path: str, target_sr: int = 22050):
        self.dataset_path = Path(dataset_path)
        self.target_sr = target_sr
        self.dataset = None
        print(f"🎵 Real Dataset Processor initialized")
        print(f"📁 Dataset path: {self.dataset_path}")
        print(f"🔊 Target sample rate: {self.target_sr}Hz")
    
    def load_dataset(self):
        """Load the HuggingFace dataset from disk"""
        try:
            print(f"📖 Loading dataset from {self.dataset_path}")
            self.dataset = load_from_disk(str(self.dataset_path))
            print(f"✅ Dataset loaded successfully!")
            print(f"   Total examples: {len(self.dataset['train'])}")
            
            # Check label distribution
            labels = self.dataset['train']['label']
            real_count = sum(1 for l in labels if l == 0)
            fake_count = sum(1 for l in labels if l == 1)
            print(f"   Real voices: {real_count}")
            print(f"   Fake voices: {fake_count}")
            print(f"   Balance ratio: {fake_count/real_count:.2f}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def process_audio_sample(self, audio_data: dict) -> Optional[np.ndarray]:
        """Process a single audio sample"""
        try:
            # Extract audio array and original sample rate
            audio_array = np.array(audio_data['array'], dtype=np.float32)
            original_sr = audio_data['sampling_rate']
            
            # Resample if needed
            if original_sr != self.target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=self.target_sr)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Ensure minimum length (1 second)
            min_length = self.target_sr
            if len(audio_array) < min_length:
                audio_array = np.pad(audio_array, (0, min_length - len(audio_array)))
            
            # Limit maximum length (10 seconds to keep processing reasonable)
            max_length = self.target_sr * 10
            if len(audio_array) > max_length:
                audio_array = audio_array[:max_length]
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing audio sample: {e}")
            return None

class VoiceFeatureExtractor:
    """Extract acoustic features from voice samples"""
    
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        print(f"🔧 Feature Extractor initialized (SR: {target_sr}Hz)")
    
    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract comprehensive acoustic features - EXACTLY matching the production system"""
        try:
            features = []
            
            # 1. MFCC features (mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
            features.extend([
                np.mean(mfcc, axis=1),  # Mean
                np.std(mfcc, axis=1),   # Standard deviation
                np.min(mfcc, axis=1),   # Minimum
                np.max(mfcc, axis=1),   # Maximum
            ])
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.target_sr)
            
            features.extend([
                [np.mean(spectral_centroids), np.std(spectral_centroids), np.min(spectral_centroids), np.max(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff), np.min(spectral_rolloff), np.max(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.min(spectral_bandwidth), np.max(spectral_bandwidth)],
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1)
            ])
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append([np.mean(zcr), np.std(zcr), np.min(zcr), np.max(zcr)])
            
            # 4. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # 5. Mel-scale spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.target_sr, n_mels=13)
            mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            features.extend([
                np.mean(mel_db, axis=1),
                np.std(mel_db, axis=1)
            ])
            
            # 6. Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            total_energy = np.sum(audio**2)
            
            features.append([
                harmonic_energy / (total_energy + 1e-10),
                percussive_energy / (total_energy + 1e-10),
                harmonic_energy / (percussive_energy + 1e-10)
            ])
            
            # 7. Tempo and rhythm
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=self.target_sr)
                features.append([tempo])
            except:
                features.append([120.0])  # Default tempo
            
            # 8. RMS energy
            rms = librosa.feature.rms(y=audio)
            features.append([np.mean(rms), np.std(rms), np.min(rms), np.max(rms)])
            
            # 9. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features.append([np.mean(spectral_flatness), np.std(spectral_flatness)])
            
            # 10. Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=self.target_sr)
            features.extend([
                np.mean(tonnetz, axis=1),
                np.std(tonnetz, axis=1)
            ])
            
            # Flatten all features
            flat_features = []
            for feature_group in features:
                if np.isscalar(feature_group):
                    flat_features.append(feature_group)
                elif hasattr(feature_group, 'flatten'):
                    flat_features.extend(feature_group.flatten())
                else:
                    flat_features.extend(np.array(feature_group).flatten())
            
            return np.array(flat_features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

class RealDatasetTrainer:
    """Train voice clone detection models using the real dataset"""
    
    def __init__(self, dataset_path: str, sample_limit: Optional[int] = None):
        self.dataset_path = dataset_path
        self.sample_limit = sample_limit  # Limit samples for faster training if needed
        self.processor = RealDatasetProcessor(dataset_path)
        self.feature_extractor = VoiceFeatureExtractor()
        self.models = {}
        self.scaler = StandardScaler()
        self.training_stats = {}
        
        print(f"🚀 Real Dataset Trainer initialized")
        if sample_limit:
            print(f"⚡ Sample limit set to: {sample_limit}")
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from the real dataset"""
        print(f"📊 Preparing training data...")
        
        # Load dataset
        if not self.processor.load_dataset():
            raise Exception("Failed to load dataset")
        
        features_list = []
        labels_list = []
        sources_list = []
        
        # Get dataset
        dataset = self.processor.dataset['train']
        
        # Apply sample limit if specified
        total_samples = len(dataset)
        if self.sample_limit and self.sample_limit < total_samples:
            print(f"🔄 Using {self.sample_limit} samples out of {total_samples}")
            # Create a balanced subset
            real_indices = [i for i, label in enumerate(dataset['label']) if label == 0]
            fake_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
            
            samples_per_class = self.sample_limit // 2
            real_subset = real_indices[:min(samples_per_class, len(real_indices))]
            fake_subset = fake_indices[:min(samples_per_class, len(fake_indices))]
            
            selected_indices = real_subset + fake_subset
            np.random.shuffle(selected_indices)
            
            # Filter dataset
            dataset = dataset.select(selected_indices)
            print(f"✅ Selected balanced subset: {len(dataset)} samples")
        
        print(f"🔊 Processing {len(dataset)} audio samples...")
        
        # Process each sample
        successful_samples = 0
        failed_samples = 0
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing audio")):
            try:
                # Process audio
                audio = self.processor.process_audio_sample(sample['audio'])
                if audio is None:
                    failed_samples += 1
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(audio)
                if features is None:
                    failed_samples += 1
                    continue
                
                features_list.append(features)
                labels_list.append(sample['label'])  # 0=real, 1=fake
                sources_list.append(sample.get('source', 'unknown'))
                successful_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                failed_samples += 1
                continue
        
        print(f"✅ Successfully processed: {successful_samples} samples")
        print(f"❌ Failed to process: {failed_samples} samples")
        
        if successful_samples == 0:
            raise Exception("No samples were successfully processed!")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"📊 Final dataset shape:")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {y.shape}")
        print(f"   Real samples: {np.sum(y == 0)}")
        print(f"   Fake samples: {np.sum(y == 1)}")
        print(f"   Feature dimensions: {X.shape[1]}")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models on the dataset"""
        print(f"🎯 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print(f"⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'description': 'XGBoost Classifier'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'description': 'Random Forest Classifier'
            },
            'svm': {
                'model': SVC(
                    kernel='rbf',
                    C=10.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                ),
                'description': 'Support Vector Machine'
            }
        }
        
        # Train each model
        for name, config in models_config.items():
            print(f"\n🔥 Training {config['description']}...")
            
            start_time = datetime.now()
            model = config['model']
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            self.models[name] = model
            self.training_stats[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'training_time': training_time,
                'test_predictions': y_pred,
                'test_probabilities': y_pred_proba
            }
            
            print(f"✅ {config['description']} Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC Score: {auc_score:.4f}")
            print(f"   Training Time: {training_time:.2f}s")
        
        # Print detailed classification reports
        print(f"\n📊 Detailed Results:")
        for name, stats in self.training_stats.items():
            print(f"\n{name.upper()} Classification Report:")
            print(classification_report(y_test, stats['test_predictions'], 
                                      target_names=['Real', 'Fake']))
    
    def save_models(self, output_dir: str = "trained_models"):
        """Save trained models and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 Saving models to {output_path}")
        
        # Save each model
        for name, model in self.models.items():
            model_file = output_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"✅ Saved {name} model")
        
        # Save scaler
        scaler_file = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        print(f"✅ Saved feature scaler")
        
        # Find best model
        best_model = max(self.training_stats.items(), key=lambda x: x[1]['accuracy'])
        
        # Save metadata
        metadata = {
            'best_model': best_model[0],
            'models': {name: {
                'accuracy': stats['accuracy'],
                'auc_score': stats['auc_score'],
                'training_time': stats['training_time']
            } for name, stats in self.training_stats.items()},
            'feature_extractor_type': 'Real_Dataset_Comprehensive_Acoustic',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_features': len(self.feature_extractor.extract_features(np.random.randn(22050))),
            'dataset_source': 'HuggingFace_ArissBandoss_fake_or_real_dataset',
            'sample_rate': self.feature_extractor.target_sr
        }
        
        metadata_file = output_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata")
        print(f"🏆 Best model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

def main():
    """Main training function"""
    print("="*60)
    print("🎯 VOICE CLONE DETECTION - REAL DATASET TRAINING")
    print("="*60)
    
    # Configuration
    dataset_path = "Voice_Dataset/fake_or_real_dataset"
    sample_limit = None  # Use None for full dataset, or set a number like 5000 for faster testing
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset has been downloaded first.")
        return
    
    try:
        # Initialize trainer
        trainer = RealDatasetTrainer(dataset_path, sample_limit=sample_limit)
        
        # Prepare training data
        X, y = trainer.prepare_training_data()
        
        # Train models
        trainer.train_models(X, y)
        
        # Save models
        trainer.save_models()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your voice clone detection models are now trained on real data")
        print("and ready to accurately detect cloned voices!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
