"""
URL configuration for ghost project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from core import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('deepfake-demo/', views.deepfake_demo, name='deepfake_demo'),
    path('fraud-demo/', views.fraud_demo, name='fraud_demo'),
    path('otp-demo/', views.otp_demo, name='otp_demo'),
    path('otp-demo-fixed/', views.otp_demo_fixed, name='otp_demo_fixed'),
    path('simple-otp-demo/', views.simple_otp_demo, name='simple_otp_demo'),
    path('voice-demo/', views.voice_demo, name='voice_demo'),
    path('business-model/', views.business_model, name='business_model'),
    path('pricing/', views.pricing, name='pricing'),
    path('history/', views.history, name='history'),
    path('export/history/pdf/', views.export_history_pdf, name='export_history_pdf'),
    path('export/history/excel/', views.export_history_excel, name='export_history_excel'),
    # API endpoints
    path('api/log-detection/', views.api_log_detection, name='api_log_detection'),
    path('api/detection/<uuid:request_id>/', views.api_get_detection_details, name='api_get_detection_details'),
    path('api/detection/<uuid:request_id>/pdf/', views.api_download_individual_report, name='api_download_individual_report'),
    
    # Deepfake specific endpoints
    path('api/upload-deepfake-media/', views.upload_deepfake_media, name='upload_deepfake_media'),
    path('api/analyze-deepfake-media/', views.analyze_deepfake_media, name='analyze_deepfake_media'),
    path('api/deepfake-history/', views.get_deepfake_history, name='get_deepfake_history'),
    
    # Fraud detection endpoints
    path('api/upload-fraud-data/', views.upload_fraud_data, name='upload_fraud_data'),
    path('api/analyze-fraud-data/', views.analyze_fraud_data, name='analyze_fraud_data'),
    
    # Voice authentication endpoints
    path('api/upload-voice-data/', views.upload_voice_data, name='upload_voice_data'),
    path('api/analyze-voice-data/', views.analyze_voice_data, name='analyze_voice_data'),
    path('api/voice-history/', views.get_voice_history, name='get_voice_history'),
    path('api/voice-system-status/', views.voice_system_status, name='voice_system_status'),
    path('api/voice-detection-history/', views.voice_detection_history, name='voice_detection_history'),
    
    # OTP verification endpoints
    path('api/upload-otp-data/', views.upload_otp_data, name='upload_otp_data'),
    path('api/analyze-otp-data/', views.analyze_otp_data, name='analyze_otp_data'),
    path('api/otp-history/', views.get_otp_history, name='get_otp_history'),
    
    # Voice Back OTP System endpoints
    path('api/generate-otp/', views.generate_otp, name='generate_otp'),
    path('api/verify-voice-otp/', views.verify_voice_otp, name='verify_voice_otp'),
    path('api/voice-otp-history/', views.get_voice_otp_history, name='get_voice_otp_history'),
    path('api/test-voice-recognition/', views.test_voice_recognition, name='test_voice_recognition'),
    
    # Unified detection history API
    path('api/all-detection-history/', views.get_all_detection_history, name='get_all_detection_history'),
    
    # Dashboard stats API
    path('api/dashboard-stats/', views.get_dashboard_stats, name='get_dashboard_stats'),
    
    # Real-time updates APIs
    path('api/recent-history/', views.get_recent_history_api, name='get_recent_history_api'),
    path('api/dashboard-stats-realtime/', views.get_dashboard_stats_api, name='get_dashboard_stats_api'),
    
    # Report download endpoints
    path('api/report/download/<int:log_id>/<str:report_type>/', views.download_analysis_report, name='download_analysis_report'),
    path('api/report/download/latest/voice/', views.download_latest_voice_report, name='download_latest_voice_report'),
    path('api/report/bulk/<str:report_type>/', views.generate_bulk_report, name='generate_bulk_report'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
