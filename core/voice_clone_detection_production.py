"""
Production Voice Clone Detection System
=====================================

This module provides production-ready voice clone detection using
trained machine learning models with comprehensive feature extraction.

Author: SAP GHOST AI Team
Version: 4.0 - Production Ready
"""

import os
import json
import time
import tempfile
import numpy as np
import librosa
import soundfile as sf
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class VoiceFeatureExtractor:
    """Extract comprehensive acoustic features from audio"""
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        
    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, duration=30)
            return self.preprocess_audio(audio, sr)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Preprocess raw audio data"""
        try:
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract comprehensive acoustic features"""
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


class VoiceCloneDetectionProduction:
    """Production-ready voice clone detection system"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            # Default to trained_models directory in project root
            current_dir = Path(__file__).parent.parent
            self.models_dir = current_dir / "trained_models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models = {}
        self.scaler = None
        self.metadata = {}
        self.feature_extractor = VoiceFeatureExtractor()
        self.is_initialized = False
        
        print(f"🚀 Initializing Production Voice Clone Detection System")
        print(f"📁 Models directory: {self.models_dir}")
    
    def models_available(self) -> bool:
        """Check if required model files are available"""
        required_files = [
            "xgboost_model.pkl",
            "random_forest_model.pkl", 
            "svm_model.pkl",
            "scaler.pkl"
        ]
        
        return all((self.models_dir / f).exists() for f in required_files)
    
    def initialize(self) -> bool:
        """Initialize the detection system by loading models"""
        try:
            if not self.models_dir.exists():
                print(f"❌ Models directory not found: {self.models_dir}")
                return False
            
            if not self.models_available():
                print("❌ Required model files not found!")
                print("Please run the training script first: python train_simple_voice_detector.py")
                return False
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded model metadata")
                print(f"   Best model: {self.metadata.get('best_model', 'Unknown')}")
                print(f"   Training date: {self.metadata.get('training_date', 'Unknown')}")
            else:
                print("⚠️  No metadata found, using default settings")
                self.metadata = {'best_model': 'xgboost'}
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded feature scaler")
            else:
                print("❌ Scaler not found!")
                return False
            
            # Load models
            model_files = {
                'xgboost': 'xgboost_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'svm': 'svm_model.pkl'
            }
            
            loaded_models = 0
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    try:
                        self.models[name] = joblib.load(model_path)
                        print(f"✅ Loaded {name} model")
                        loaded_models += 1
                    except Exception as e:
                        print(f"❌ Error loading {name} model: {e}")
                else:
                    print(f"⚠️  {name} model not found")
            
            if loaded_models == 0:
                print("❌ No models loaded successfully!")
                return False
            
            self.is_initialized = True
            print(f"✅ Production Voice Clone Detection System initialized!")
            print(f"   Models loaded: {loaded_models}")
            print(f"   Feature dimensions: {getattr(self.scaler, 'n_features_in_', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing detection system: {e}")
            return False
    
    def _predict_with_model(self, features: np.ndarray, model_name: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Make prediction with a specific model"""
        try:
            if model_name not in self.models:
                return None, None
                
            model = self.models[model_name]
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            return int(prediction), probability
            
        except Exception as e:
            print(f"Error making prediction with {model_name}: {e}")
            return None, None
    
    def predict_file(self, audio_path: str) -> Dict[str, Any]:
        """Analyze voice file for clone detection"""
        start_time = time.time()
        
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize detection system',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
        
        try:
            print(f"🔍 Analyzing voice file: {audio_path}")
            
            # Load and preprocess audio
            audio = self.feature_extractor.load_audio_file(audio_path)
            if audio is None:
                return {
                    'success': False,
                    'error': 'Failed to load audio file',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
            
            audio_duration = len(audio) / self.feature_extractor.target_sr
            
            # Extract features
            print("🔧 Extracting features...")
            features = self.feature_extractor.extract_features(audio)
            if features is None:
                return {
                    'success': False,
                    'error': 'Failed to extract features',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
            
            print(f"✅ Extracted {len(features)} features")
            
            # Make predictions with all available models
            print("🤖 Running model predictions...")
            predictions = {}
            probabilities = {}
            
            for model_name in self.models.keys():
                pred, prob = self._predict_with_model(features, model_name)
                if pred is not None:
                    predictions[model_name] = pred
                    probabilities[model_name] = {
                        'real': float(prob[0]),
                        'fake': float(prob[1]),
                        'confidence': float(max(prob)) * 100
                    }
            
            if not predictions:
                return {
                    'success': False,
                    'error': 'All model predictions failed',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
            
            # Use best model for final decision
            best_model = self.metadata.get('best_model', 'xgboost')
            if best_model not in predictions:
                best_model = list(predictions.keys())[0]  # Use first available model
            
            final_prediction = predictions[best_model]
            final_probability = probabilities[best_model]
            
            # Calculate ensemble prediction (majority vote)
            fake_votes = sum(1 for pred in predictions.values() if pred == 1)
            real_votes = len(predictions) - fake_votes
            ensemble_prediction = 1 if fake_votes > real_votes else 0
            
            # Calculate confidence
            confidence = final_probability['confidence']
            
            result = 'fake' if final_prediction == 1 else 'real'
            ensemble_result = 'fake' if ensemble_prediction == 1 else 'real'
            
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'classification': {
                    'result': result,
                    'confidence': confidence,
                    'fake_probability': final_probability['fake'] * 100,
                    'real_probability': final_probability['real'] * 100
                },
                'ensemble_result': {
                    'result': ensemble_result,
                    'confidence': max([p['confidence'] for p in probabilities.values()]),
                    'verdict': 'CLONED' if ensemble_result == 'fake' else 'AUTHENTIC'
                },
                'model_predictions': {
                    model: {
                        'prediction': 'fake' if pred == 1 else 'real',
                        'confidence': probabilities[model]['confidence'],
                        'probabilities': probabilities[model]
                    }
                    for model, pred in predictions.items()
                },
                'audio_info': {
                    'duration': float(audio_duration),
                    'sample_rate': int(self.feature_extractor.target_sr),
                    'file_path': audio_path
                },
                'processing_info': {
                    'processing_time': float(processing_time),
                    'best_model_used': best_model,
                    'models_available': list(self.models.keys()),
                    'feature_count': len(features),
                    'version': '4.0_Production'
                }
            }
            
            print(f"✅ Analysis complete! Result: {result.upper()}")
            print(f"   Ensemble Result: {ensemble_result.upper()}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Processing time: {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            error_msg = f"Error during voice analysis: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'classification': {'result': 'error', 'confidence': 0.0},
                'processing_info': {'processing_time': time.time() - start_time}
            }
    
    def predict_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze raw audio data for clone detection"""
        start_time = time.time()
        
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize detection system',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
        
        try:
            print("🔍 Analyzing raw audio data")
            
            # Preprocess audio data
            audio = self.feature_extractor.preprocess_audio(audio_data, sample_rate)
            if audio is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess audio data',
                    'classification': {'result': 'error', 'confidence': 0.0}
                }
            
            # Save to temporary file and analyze
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.feature_extractor.target_sr)
                temp_path = temp_file.name
            
            try:
                # Analyze using the file-based method
                results = self.predict_file(temp_path)
                if 'audio_info' in results:
                    results['audio_info']['source'] = 'raw_data'
                return results
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"Error analyzing raw audio data: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'classification': {'result': 'error', 'confidence': 0.0},
                'processing_info': {'processing_time': time.time() - start_time}
            }


# Test function
def test_production_detector():
    """Test the production detector"""
    print("🧪 Testing Production Voice Clone Detection System...")
    
    # Create test audio
    fs = 22050
    duration = 3
    t = np.linspace(0, duration, fs * duration)
    # Create more natural sounding test audio
    audio = 0.3 * (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t))
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # Save test file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, fs)
        test_file_path = f.name
    
    try:
        # Test the detector
        detector = VoiceCloneDetectionProduction()
        results = detector.predict_file(test_file_path)
        
        print("\n📊 Test Results:")
        print(json.dumps(results, indent=2))
        
        return results
    finally:
        # Clean up
        try:
            os.unlink(test_file_path)
        except:
            pass


if __name__ == "__main__":
    test_production_detector()
