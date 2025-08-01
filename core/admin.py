from django.contrib import admin
from .models import UserProfile, DeepfakeDetectionLog, FraudDetectionLog, VoiceDetectionLog

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'otp_enabled', 'created_at')
    search_fields = ('user__username', 'phone_number')

@admin.register(DeepfakeDetectionLog)
class DeepfakeDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_fake', 'confidence_score', 'created_at')
    list_filter = ('is_fake', 'created_at')
    search_fields = ('user__username',)

@admin.register(FraudDetectionLog)
class FraudDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_fraudulent', 'risk_score', 'created_at')
    list_filter = ('is_fraudulent', 'created_at')
    search_fields = ('user__username',)

@admin.register(VoiceDetectionLog)
class VoiceDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'submission_type', 'result', 'confidence_score', 'audio_duration', 'created_at')
    list_filter = ('submission_type', 'result', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at', 'processing_time')
