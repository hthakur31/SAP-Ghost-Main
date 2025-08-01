"""
Voice Clone Detection Model Utilities
=====================================

Utility functions and classes for the voice clone detection system.
Handles model loading, feature caching, and system health monitoring.

Author: SAP GHOST AI Team
Version: 3.0
"""

import os
import json
import pickle
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
import numpy as np

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Caching system for model predictions and features
    """
    
    def __init__(self, cache_ttl: int = 3600):  # 1 hour cache
        self.cache_ttl = cache_ttl
        self.prefix = "voice_detection_"
    
    def _get_audio_hash(self, audio_data: np.ndarray) -> str:
        """
        Generate hash for audio data
        """
        return hashlib.md5(audio_data.tobytes()).hexdigest()
    
    def get_cached_result(self, audio_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result
        """
        try:
            cache_key = f"{self.prefix}result_{audio_hash}"
            return cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    def cache_result(self, audio_hash: str, result: Dict[str, Any]):
        """
        Cache analysis result
        """
        try:
            cache_key = f"{self.prefix}result_{audio_hash}"
            cache.set(cache_key, result, self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def get_cached_features(self, audio_hash: str) -> Optional[np.ndarray]:
        """
        Get cached feature vector
        """
        try:
            cache_key = f"{self.prefix}features_{audio_hash}"
            cached_data = cache.get(cache_key)
            if cached_data:
                return np.frombuffer(cached_data, dtype=np.float32)
            return None
        except Exception as e:
            logger.warning(f"Feature cache get failed: {e}")
            return None
    
    def cache_features(self, audio_hash: str, features: np.ndarray):
        """
        Cache feature vector
        """
        try:
            cache_key = f"{self.prefix}features_{audio_hash}"
            cache.set(cache_key, features.tobytes(), self.cache_ttl)
        except Exception as e:
            logger.warning(f"Feature cache set failed: {e}")


class SystemHealthMonitor:
    """
    Monitor system health and performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'model_load_status': {
                'ecapa_model': False,
                'svm_classifier': False,
                'mlp_classifier': False
            }
        }
    
    def record_analysis(self, success: bool, processing_time: float):
        """
        Record analysis metrics
        """
        self.metrics['total_analyses'] += 1
        
        if success:
            self.metrics['successful_analyses'] += 1
        else:
            self.metrics['failed_analyses'] += 1
        
        # Update rolling average
        total_time = self.metrics['average_processing_time'] * (self.metrics['total_analyses'] - 1)
        self.metrics['average_processing_time'] = (total_time + processing_time) / self.metrics['total_analyses']
    
    def update_model_status(self, model_name: str, status: bool):
        """
        Update model loading status
        """
        if model_name in self.metrics['model_load_status']:
            self.metrics['model_load_status'][model_name] = status
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get system health report
        """
        if self.metrics['total_analyses'] > 0:
            success_rate = (self.metrics['successful_analyses'] / self.metrics['total_analyses']) * 100
        else:
            success_rate = 0.0
        
        return {
            'overall_health': 'good' if success_rate > 90 else 'warning' if success_rate > 70 else 'critical',
            'success_rate': success_rate,
            'total_analyses': self.metrics['total_analyses'],
            'average_processing_time': self.metrics['average_processing_time'],
            'model_status': self.metrics['model_load_status'],
            'last_updated': timezone.now().isoformat()
        }


class AudioFileHandler:
    """
    Handle audio file operations with validation and conversion
    """
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_DURATION = 300  # 5 minutes
    
    @classmethod
    def validate_audio_file(cls, file_path: str) -> Tuple[bool, str]:
        """
        Validate audio file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file size
            if path.stat().st_size > cls.MAX_FILE_SIZE:
                return False, f"File size exceeds {cls.MAX_FILE_SIZE // (1024*1024)}MB limit"
            
            # Check file extension
            if path.suffix.lower() not in cls.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported: {', '.join(cls.SUPPORTED_FORMATS)}"
            
            # Try to load with librosa to check if it's valid audio
            try:
                import librosa
                audio, sr = librosa.load(file_path, duration=1.0)  # Load first second only for validation
                
                if len(audio) == 0:
                    return False, "Audio file appears to be empty"
                
            except Exception as e:
                return False, f"Invalid audio file: {str(e)}"
            
            return True, "Valid audio file"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    @classmethod
    def get_audio_info(cls, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information
        """
        try:
            import librosa
            
            # Load audio file
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            
            return {
                'sample_rate': sr,
                'duration': duration,
                'channels': 1,  # librosa loads as mono by default
                'file_size': Path(file_path).stat().st_size,
                'format': Path(file_path).suffix.lower(),
                'is_valid': duration > 0.5  # Minimum 0.5 seconds
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_valid': False
            }


class FeaturePipeline:
    """
    Advanced feature extraction pipeline with error handling
    """
    
    def __init__(self):
        self.feature_cache = ModelCache()
    
    def extract_robust_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features with robust error handling
        """
        features = {}
        
        # Generate audio hash for caching
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        
        # Check cache first
        cached_features = self.feature_cache.get_cached_features(audio_hash)
        if cached_features is not None:
            logger.info("Using cached features")
            return {'combined': cached_features}
        
        try:
            # Extract MFCC features
            features['mfcc'] = self._extract_mfcc_safe(audio, sr)
            
            # Extract spectral features
            features['spectral'] = self._extract_spectral_safe(audio, sr)
            
            # Extract temporal features
            features['temporal'] = self._extract_temporal_safe(audio, sr)
            
            # Combine all features
            feature_list = []
            for feat_type, feat_values in features.items():
                if feat_values is not None and len(feat_values) > 0:
                    feature_list.append(feat_values)
            
            if feature_list:
                combined_features = np.concatenate(feature_list)
                features['combined'] = combined_features
                
                # Cache the combined features
                self.feature_cache.cache_features(audio_hash, combined_features)
            else:
                # Return zero features if all extractions failed
                features['combined'] = np.zeros(500)  # Default feature size
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {'combined': np.zeros(500)}
    
    def _extract_mfcc_safe(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Safely extract MFCC features
        """
        try:
            import librosa
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Statistical features
            stats = np.array([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.min(mfcc, axis=1),
                np.max(mfcc, axis=1)
            ]).flatten()
            
            return stats
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return np.zeros(52)  # 13 * 4 statistics
    
    def _extract_spectral_safe(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Safely extract spectral features
        """
        try:
            import librosa
            
            features = []
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.extend([np.mean(centroid), np.std(centroid)])
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.extend([np.mean(rolloff), np.std(rolloff)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.extend([np.mean(bandwidth), np.std(bandwidth)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return np.zeros(10)
    
    def _extract_temporal_safe(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Safely extract temporal features
        """
        try:
            import librosa
            
            features = []
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            features.extend([np.mean(rms), np.std(rms)])
            
            # Spectral flux
            stft = librosa.stft(audio)
            spectral_flux = np.sum(np.diff(np.abs(stft), axis=1)**2, axis=0)
            features.extend([np.mean(spectral_flux), np.std(spectral_flux)])
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Temporal feature extraction failed: {e}")
            return np.zeros(5)


class ResultFormatter:
    """
    Format analysis results for different consumers
    """
    
    @staticmethod
    def format_for_frontend(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results for frontend consumption
        """
        try:
            classification = results.get('classification', {})
            language_analysis = results.get('language_analysis', {})
            
            return {
                'status': 'success',
                'result': classification.get('result', 'unknown'),
                'confidence': round(classification.get('confidence_score', 0.0), 1),
                'is_cloned': classification.get('result') == 'cloned',
                'language': {
                    'detected': language_analysis.get('language_name', 'Unknown'),
                    'confidence': round(language_analysis.get('confidence', 0.0), 1)
                },
                'processing_info': {
                    'duration': round(results.get('audio_info', {}).get('duration', 0.0), 2),
                    'processing_time': round(results.get('processing_info', {}).get('processing_time', 0.0), 2)
                },
                'advanced_metrics': results.get('advanced_metrics', {}),
                'timestamp': results.get('processing_info', {}).get('timestamp', timezone.now().timestamp())
            }
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'confidence': 0.0
            }
    
    @staticmethod
    def format_for_database(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results for database storage
        """
        try:
            return {
                'classification_result': results.get('classification', {}).get('result', 'unknown'),
                'confidence_score': results.get('classification', {}).get('confidence_score', 0.0),
                'language_detected': results.get('language_analysis', {}).get('detected_language', 'unknown'),
                'language_confidence': results.get('language_analysis', {}).get('confidence', 0.0),
                'processing_time': results.get('processing_info', {}).get('processing_time', 0.0),
                'audio_duration': results.get('audio_info', {}).get('duration', 0.0),
                'model_version': results.get('processing_info', {}).get('model_version', '3.0'),
                'feature_count': results.get('feature_analysis', {}).get('total_features', 0),
                'analysis_timestamp': timezone.now().isoformat(),
                'full_results': json.dumps(results, default=str)
            }
            
        except Exception as e:
            logger.error(f"Database formatting failed: {e}")
            return {
                'classification_result': 'error',
                'confidence_score': 0.0,
                'error_message': str(e)
            }


# Global instances
model_cache = ModelCache()
health_monitor = SystemHealthMonitor()
feature_pipeline = FeaturePipeline()
result_formatter = ResultFormatter()
