from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class UserProfile(models.Model):
    SUBSCRIPTION_TIERS = [
        ('FREEMIUM', 'Freemium'),
        ('PREMIUM', 'Premium'),
        ('ENTERPRISE', 'Enterprise'),
    ]
    
    ROLE_CHOICES = [
        ('USER', 'User'),
        ('ADMIN', 'Admin'),
        ('MANAGER', 'Manager'),
        ('ANALYST', 'Analyst'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, blank=True, default='')
    company = models.CharField(max_length=100, blank=True, default='')
    role = models.CharField(max_length=50, choices=ROLE_CHOICES, default='USER')
    subscription_tier = models.CharField(max_length=20, choices=SUBSCRIPTION_TIERS, default='FREEMIUM')
    otp_enabled = models.BooleanField(default=False)
    voice_print = models.FileField(upload_to='voice_prints/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} ({self.get_subscription_tier_display()})"

class DetectionLog(models.Model):
    DETECTION_TYPES = [
        ('deepfake', 'Deepfake Detection'),
        ('voice', 'Voice Authentication'),
        ('fraud', 'Fraud Detection'),
        ('otp', 'OTP Verification'),
    ]
    
    RESULT_CHOICES = [
        ('verified', 'Verified'),
        ('suspicious', 'Suspicious'),
        ('authentic', 'Authentic'),
        ('fraudulent', 'Fraudulent'),
        ('rejected', 'Rejected'),
    ]
    
    SUBSCRIPTION_PLANS = [
        ('FREEMIUM', 'Freemium'),
        ('PREMIUM', 'Premium'),
        ('ENTERPRISE', 'Enterprise'),
    ]

    request_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    analysis_type = models.CharField(max_length=20, choices=DETECTION_TYPES)
    result = models.CharField(max_length=20, choices=RESULT_CHOICES)
    confidence = models.FloatField(help_text="Confidence percentage (0-100)")
    subscription_plan = models.CharField(max_length=20, choices=SUBSCRIPTION_PLANS, default='FREEMIUM')
    
    # File references
    uploaded_file = models.FileField(upload_to='detection_files/', null=True, blank=True)
    result_file = models.FileField(upload_to='result_files/', null=True, blank=True)
    
    # Additional data
    metadata = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Detection Log'
        verbose_name_plural = 'Detection Logs'

    def __str__(self):
        return f"{self.get_analysis_type_display()} - {self.get_result_display()} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    @property
    def is_threat_detected(self):
        return self.result in ['suspicious', 'fraudulent', 'rejected']

class DeepfakeDetectionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='deepfake_checks/', null=True, blank=True)
    video = models.FileField(upload_to='deepfake_videos/', null=True, blank=True)
    file_type = models.CharField(max_length=10, choices=[('image', 'Image'), ('video', 'Video')], default='image')
    source_type = models.CharField(max_length=20, choices=[('upload', 'Upload'), ('screen_record', 'Screen Recording')], default='upload')
    is_fake = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    analysis_details = models.JSONField(default=dict, blank=True)
    processing_time = models.FloatField(default=0.0)  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Deepfake Detection Log'
        verbose_name_plural = 'Deepfake Detection Logs'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.get_file_type_display()} - {'Fake' if self.is_fake else 'Real'} - {self.created_at}"

    @property
    def media_file(self):
        """Return the appropriate media file (image or video)"""
        return self.video if self.file_type == 'video' else self.image

class VoiceDetectionLog(models.Model):
    SUBMISSION_TYPES = [
        ('live', 'Live Recording'),
        ('upload', 'File Upload'),
    ]
    
    RESULT_CHOICES = [
        ('authentic', 'Authentic'),
        ('cloned', 'Cloned/Fake'),
        ('suspicious', 'Suspicious'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_file = models.FileField(upload_to='voice_analysis/', null=True, blank=True)
    submission_type = models.CharField(max_length=10, choices=SUBMISSION_TYPES, default='upload')
    is_cloned = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    result = models.CharField(max_length=20, choices=RESULT_CHOICES, default='authentic')
    
    # Enhanced analysis details
    analysis_details = models.JSONField(default=dict, blank=True)
    audio_duration = models.FloatField(default=0.0)  # in seconds
    processing_time = models.FloatField(default=0.0)  # in seconds
    
    # Language analysis fields
    detected_language = models.CharField(max_length=50, default='Unknown')
    language_confidence = models.FloatField(default=0.0)
    language_analysis = models.JSONField(default=dict, blank=True)
    
    # Analysis status and metadata
    analysis_type = models.CharField(max_length=50, default='voice_clone_detection')
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ])
    
    # Audio metadata
    sample_rate = models.IntegerField(default=44100)
    bit_rate = models.IntegerField(default=128)
    format = models.CharField(max_length=10, default='wav')
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Voice Detection Log'
        verbose_name_plural = 'Voice Detection Logs'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.get_result_display()} - {self.created_at}"

    @property
    def verdict(self):
        """Return user-friendly verdict"""
        if self.result == 'authentic':
            return 'Authentic Voice'
        elif self.result == 'cloned':
            return 'Cloned/Fake Voice'
        else:
            return 'Suspicious Voice'

class FraudDetectionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    transaction_data = models.JSONField(default=dict)
    is_fraudulent = models.BooleanField(default=False)
    risk_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Fraud Detection Log'
        verbose_name_plural = 'Fraud Detection Logs'

    def __str__(self):
        return f"{self.user.username} - {'Fraudulent' if self.is_fraudulent else 'Legitimate'} - {self.created_at}"
