from django.contrib.auth.models import User
from django.utils import timezone
from .models import DetectionLog
import uuid
import random

def log_detection_activity(user=None, analysis_type='deepfake', result='verified', confidence=95.0, 
                          subscription_plan='FREEMIUM', metadata=None, request=None):
    """
    Log a detection activity to the database
    
    Args:
        user: User object (optional)
        analysis_type: Type of analysis ('deepfake', 'voice', 'fraud', 'otp')
        result: Result of detection ('verified', 'suspicious', 'authentic', 'fraudulent', 'rejected')
        confidence: Confidence percentage (0-100)
        subscription_plan: User's subscription plan
        metadata: Additional data as dict
        request: HTTP request object for IP and user agent
    """
    
    # Get IP and user agent from request
    ip_address = None
    user_agent = None
    if request:
        ip_address = get_client_ip(request)
        user_agent = request.META.get('HTTP_USER_AGENT', '')
    
    # Create log entry
    log_entry = DetectionLog.objects.create(
        user=user,
        analysis_type=analysis_type,
        result=result,
        confidence=confidence,
        subscription_plan=subscription_plan,
        metadata=metadata or {},
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    return log_entry

def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_deepfake_detection(request, is_fake=False, confidence=95.0, file_info=None):
    """Log deepfake detection activity"""
    result = 'suspicious' if is_fake else 'authentic'
    metadata = {
        'file_type': 'image',
        'detection_method': 'CNN_analysis',
        'file_info': file_info or {}
    }
    
    return log_detection_activity(
        user=request.user if request.user.is_authenticated else None,
        analysis_type='deepfake',
        result=result,
        confidence=confidence,
        metadata=metadata,
        request=request
    )

def log_voice_authentication(request, is_verified=True, confidence=92.0, voice_data=None):
    """Log voice authentication activity"""
    result = 'verified' if is_verified else 'rejected'
    metadata = {
        'voice_duration': voice_data.get('duration', 0) if voice_data else 0,
        'audio_quality': voice_data.get('quality', 'good') if voice_data else 'good',
        'voice_features': voice_data or {}
    }
    
    return log_detection_activity(
        user=request.user if request.user.is_authenticated else None,
        analysis_type='voice',
        result=result,
        confidence=confidence,
        metadata=metadata,
        request=request
    )

def log_fraud_detection(request, is_fraudulent=False, confidence=88.0, transaction_data=None):
    """Log fraud detection activity"""
    result = 'fraudulent' if is_fraudulent else 'verified'
    metadata = {
        'transaction_amount': transaction_data.get('amount', 0) if transaction_data else 0,
        'merchant': transaction_data.get('merchant', 'Unknown') if transaction_data else 'Unknown',
        'risk_factors': transaction_data or {}
    }
    
    return log_detection_activity(
        user=request.user if request.user.is_authenticated else None,
        analysis_type='fraud',
        result=result,
        confidence=confidence,
        metadata=metadata,
        request=request
    )

def log_otp_verification(request, is_verified=True, confidence=97.0, otp_data=None):
    """Log OTP verification activity"""
    result = 'verified' if is_verified else 'rejected'
    metadata = {
        'otp_method': 'voice_back',
        'attempts': otp_data.get('attempts', 1) if otp_data else 1,
        'verification_data': otp_data or {}
    }
    
    return log_detection_activity(
        user=request.user if request.user.is_authenticated else None,
        analysis_type='otp',
        result=result,
        confidence=confidence,
        metadata=metadata,
        request=request
    )

# Demo functions that can be called from views
def demo_deepfake_detection(request):
    """Simulate deepfake detection for demo"""
    # Random result for demo
    is_fake = random.choice([True, False])
    confidence = random.uniform(75.0, 99.0)
    
    return log_deepfake_detection(
        request=request,
        is_fake=is_fake,
        confidence=confidence,
        file_info={'demo': True, 'file_name': 'demo_image.jpg'}
    )

def demo_voice_authentication(request):
    """Simulate voice authentication for demo"""
    is_verified = random.choice([True, False])
    confidence = random.uniform(80.0, 98.0)
    
    return log_voice_authentication(
        request=request,
        is_verified=is_verified,
        confidence=confidence,
        voice_data={'demo': True, 'duration': random.randint(3, 10)}
    )

def demo_fraud_detection(request):
    """Simulate fraud detection for demo"""
    is_fraudulent = random.choice([True, False])
    confidence = random.uniform(70.0, 95.0)
    
    return log_fraud_detection(
        request=request,
        is_fraudulent=is_fraudulent,
        confidence=confidence,
        transaction_data={'demo': True, 'amount': random.uniform(10.0, 1000.0)}
    )

def demo_otp_verification(request):
    """Simulate OTP verification for demo"""
    is_verified = random.choice([True, False])
    confidence = random.uniform(85.0, 99.0)
    
    return log_otp_verification(
        request=request,
        is_verified=is_verified,
        confidence=confidence,
        otp_data={'demo': True, 'otp_code': '123456'}
    )
