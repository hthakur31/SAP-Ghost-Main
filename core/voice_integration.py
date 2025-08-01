"""
Django Integration for Voice Clone Detection System
==================================================

This module integrates the advanced AI voice clone detection system 
with Django views and models using our trained models.

Author: SAP GHOST AI Team
Version: 4.0 - Production Ready
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.http import JsonResponse
from django.utils import timezone
from .voice_clone_detection_production import VoiceCloneDetectionProduction
from .models import VoiceDetectionLog
import logging

# Setup logging
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

class VoiceCloneDetectionService:
    """
    Django service for voice clone detection using trained models
    """
    
    def __init__(self):
        self.detector = VoiceCloneDetectionProduction()
        self.is_initialized = False
        self._initialize_detector()
    
    def _initialize_detector(self):
        """
        Initialize the voice clone detector with trained models
        """
        try:
            if not self.is_initialized:
                logger.info("Initializing Voice Clone Detection System with trained models...")
                
                # Check if models exist
                if self.detector.models_available():
                    self.is_initialized = True
                    logger.info("Voice Clone Detection System initialized successfully")
                else:
                    logger.warning("Trained models not found, running in demo mode")
                    self.is_initialized = True  # Still allow operation in demo mode
                    
        except Exception as e:
            logger.error(f"Failed to initialize voice clone detector: {str(e)}")
            self.is_initialized = False
    
    def analyze_uploaded_file(self, uploaded_file: UploadedFile, user_profile=None) -> Dict[str, Any]:
        """
        Analyze an uploaded audio file for voice cloning using trained models
        
        Args:
            uploaded_file: Django uploaded file object
            user_profile: User profile for logging
            
        Returns:
            Analysis results dictionary
        """
        temp_file_path = None
        
        try:
            # Validate file
            if not self._validate_audio_file(uploaded_file):
                return {
                    'success': False,
                    'error': 'Invalid audio file format. Supported: WAV, MP3, M4A, OGG',
                    'classification': {'result': 'error', 'confidence_score': 0.0}
                }
            
            # Save uploaded file to temporary location
            temp_file_path = self._save_temp_file(uploaded_file)
            
            # Analyze the audio file using trained models
            analysis_results = self.detector.predict_file(temp_file_path)
            
            # Convert numpy types to ensure JSON serializability
            analysis_results = convert_numpy_types(analysis_results)
            
            # Normalize the response format for Django compatibility
            if 'classification' in analysis_results:
                classification = analysis_results['classification']
                # Convert 'confidence' to 'confidence_score' for Django compatibility
                if 'confidence' in classification and 'confidence_score' not in classification:
                    classification['confidence_score'] = classification['confidence']
            
            # Save results to database
            if user_profile:
                self._save_to_database(
                    uploaded_file, 
                    analysis_results, 
                    user_profile,
                    temp_file_path
                )
            
            # Add success flag
            analysis_results['success'] = True
            
            # Final conversion to ensure no numpy types remain
            return convert_numpy_types(analysis_results)
            
        except Exception as e:
            error_msg = f"Error analyzing audio file: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'classification': {
                    'result': 'error',
                    'confidence_score': 0.0
                }
            }
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def _validate_audio_file(self, uploaded_file: UploadedFile) -> bool:
        """
        Validate uploaded audio file
        """
        # Check file size (max 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            return False
        
        # Check file extension
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        return file_extension in allowed_extensions
    
    def _save_temp_file(self, uploaded_file: UploadedFile) -> str:
        """
        Save uploaded file to temporary location
        """
        # Create temporary file
        suffix = Path(uploaded_file.name).suffix.lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Write uploaded file content
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        
        temp_file.close()
        return temp_file.name
    
    def _save_to_database(self, uploaded_file: UploadedFile, results: Dict[str, Any], 
                         user_profile, temp_file_path: str):
        """
        Save analysis results to database
        """
        try:
            # Convert numpy types to Python native types
            clean_results = convert_numpy_types(results)
            
            # Extract classification results
            classification = clean_results.get('classification', {})
            result = classification.get('result', 'unknown')
            confidence_score = float(classification.get('confidence', 0.0))
            
            # Prepare analysis details
            analysis_details = {
                'model_predictions': clean_results.get('model_predictions', {}),
                'ensemble_result': clean_results.get('ensemble_result', {}),
                'feature_analysis': clean_results.get('feature_analysis', {}),
                'audio_info': clean_results.get('audio_info', {}),
                'processing_info': clean_results.get('processing_info', {})
            }
            
            # Create VoiceDetectionLog entry
            voice_log = VoiceDetectionLog.objects.create(
                user=user_profile.user,  # Use user instead of user_profile
                audio_file=uploaded_file,
                result=result,
                confidence_score=confidence_score / 100.0,  # Convert to 0-1 range
                analysis_details=analysis_details,  # Model handles JSON serialization
                detected_language='Unknown',  # Our model doesn't detect language yet
                language_confidence=0.0,
                language_analysis={},
                audio_duration=float(clean_results.get('audio_info', {}).get('duration', 0.0)),
                processing_time=float(clean_results.get('processing_info', {}).get('processing_time', 0.0)),
                status='completed',
                is_cloned=(result in ['fake', 'cloned'])
            )
            
            logger.info(f"Voice detection log saved with ID: {voice_log.id}")
            
        except Exception as e:
            logger.error(f"Failed to save voice detection results: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


class VoiceAnalysisAPI:
    """
    API endpoints for voice analysis
    """
    
    def __init__(self):
        self.service = VoiceCloneDetectionService()
    
    def upload_and_analyze(self, request, user_profile=None) -> JsonResponse:
        """
        Handle voice upload and analysis API endpoint
        """
        if request.method != 'POST':
            return JsonResponse({
                'success': False,
                'error': 'Only POST method allowed'
            }, status=405)
        
        if 'audio_file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No audio file provided'
            }, status=400)
        
        uploaded_file = request.FILES['audio_file']
        
        # Analyze the file
        results = self.service.analyze_uploaded_file(uploaded_file, user_profile)
        
        # Return appropriate status code
        status_code = 200 if results.get('success', False) else 400
        
        return JsonResponse(results, status=status_code)
    
    def get_detection_history(self, request, user_profile=None) -> JsonResponse:
        """
        Get voice detection history for user
        """
        try:
            # Get recent detections
            logs = VoiceDetectionLog.objects.filter(
                user=user_profile.user
            ).order_by('-created_at')[:20]
            
            # Format results
            history = []
            for log in logs:
                try:
                    analysis_details = json.loads(log.analysis_details) if isinstance(log.analysis_details, str) else log.analysis_details
                except:
                    analysis_details = log.analysis_details if log.analysis_details else {}
                
                history.append({
                    'id': log.id,
                    'timestamp': log.created_at.isoformat(),
                    'result': log.result,
                    'confidence': float(log.confidence_score * 100),  # Convert back to percentage
                    'language': log.detected_language,
                    'status': log.status,
                    'audio_duration': float(log.audio_duration),
                    'processing_time': float(log.processing_time)
                })
            
            return JsonResponse({
                'success': True,
                'history': history,
                'total_count': len(history)
            })
            
        except Exception as e:
            logger.error(f"Error retrieving detection history: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': 'Failed to retrieve detection history'
            }, status=500)
    
    def get_system_status(self) -> JsonResponse:
        """
        Get system status and health information
        """
        try:
            status_info = {
                'success': True,
                'system_status': {
                    'detector_initialized': self.service.is_initialized,
                    'models_loaded': self.service.detector.models_available(),
                    'supported_formats': ['WAV', 'MP3', 'M4A', 'OGG', 'FLAC', 'AAC'],
                    'max_file_size_mb': 50,
                    'processing_timeout_seconds': 300,
                    'version': '4.0',
                    'tech_stack': 'XGBoost + Random Forest + SVM + LibROSA Features'
                },
                'capabilities': {
                    'voice_clone_detection': True,
                    'ensemble_prediction': True,
                    'advanced_features': True,
                    'acoustic_analysis': True,
                    'multi_model_voting': True
                },
                'models': {
                    'xgboost': self.service.detector.models_available(),
                    'random_forest': self.service.detector.models_available(),
                    'svm': self.service.detector.models_available(),
                    'feature_scaler': self.service.detector.models_available()
                }
            }
            
            return JsonResponse(status_info)
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': 'Failed to get system status'
            }, status=500)


# Global service instance
voice_detection_service = VoiceCloneDetectionService()
voice_analysis_api = VoiceAnalysisAPI()


def analyze_voice_file(audio_file_path: str, user_profile=None) -> Dict[str, Any]:
    """
    Standalone function to analyze voice file using trained models
    
    Args:
        audio_file_path: Path to audio file
        user_profile: Optional user profile for logging
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Initialize service if needed
        if not voice_detection_service.is_initialized:
            voice_detection_service._initialize_detector()
        
        # Analyze the file using trained models
        results = voice_detection_service.detector.predict_file(audio_file_path)
        results['success'] = True
        
        return results
        
    except Exception as e:
        error_msg = f"Error analyzing voice file: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'classification': {
                'result': 'error',
                'confidence_score': 0.0
            }
        }


def get_analysis_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key information for UI display
    
    Args:
        results: Full analysis results
        
    Returns:
        Simplified summary for frontend
    """
    try:
        # Convert numpy types first
        clean_results = convert_numpy_types(results)
        
        classification = clean_results.get('classification', {})
        ensemble_result = clean_results.get('ensemble_result', {})
        model_predictions = clean_results.get('model_predictions', {})
        
        return {
            'result': classification.get('result', 'unknown'),
            'confidence': float(classification.get('confidence', 0.0)),
            'fake_probability': float(classification.get('fake_probability', 0.0)),
            'authentic_probability': float(classification.get('real_probability', 0.0)),
            'processing_time': float(clean_results.get('processing_info', {}).get('processing_time', 0.0)),
            'audio_duration': float(clean_results.get('audio_info', {}).get('duration', 0.0)),
            'model_scores': {
                'xgboost': float(model_predictions.get('xgboost', {}).get('confidence', 0.0)),
                'random_forest': float(model_predictions.get('random_forest', {}).get('confidence', 0.0)),
                'svm': float(model_predictions.get('svm', {}).get('confidence', 0.0)),
                'ensemble': float(ensemble_result.get('confidence', 0.0))
            },
            'verdict': ensemble_result.get('verdict', 'UNKNOWN').upper()
        }
        
    except Exception as e:
        logger.error(f"Error creating analysis summary: {str(e)}")
        return {
            'result': 'error',
            'confidence': 0.0,
            'error': str(e)
        }
