from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from core.models import DetectionLog
import random
import uuid
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = 'Create sample detection logs for testing'

    def handle(self, *args, **options):
        # Create a test user if it doesn't exist
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        
        if created:
            user.set_password('testpass123')
            user.save()
            self.stdout.write(f"Created test user: {user.username}")
        
        # Sample data for realistic logs
        analysis_types = ['deepfake', 'voice', 'fraud', 'otp']
        results = ['verified', 'suspicious', 'authentic', 'fraudulent', 'rejected']
        subscription_plans = ['FREEMIUM', 'PREMIUM', 'ENTERPRISE']
        
        # Create sample logs
        logs_created = 0
        for i in range(50):  # Create 50 sample logs
            # Random timestamp within last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = timezone.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Random data
            analysis_type = random.choice(analysis_types)
            result = random.choice(results)
            confidence = round(random.uniform(70.0, 99.9), 2)
            subscription_plan = random.choice(subscription_plans)
            
            # Ensure certain combinations make sense
            if analysis_type == 'deepfake' and result in ['verified', 'authentic']:
                confidence = random.uniform(85.0, 99.9)
            elif result in ['suspicious', 'fraudulent']:
                confidence = random.uniform(70.0, 85.0)
            
            # Create log entry
            DetectionLog.objects.create(
                user=user,
                timestamp=timestamp,
                analysis_type=analysis_type,
                result=result,
                confidence=confidence,
                subscription_plan=subscription_plan,
                metadata={
                    'file_size': random.randint(100000, 10000000),
                    'duration': random.randint(1, 300),
                    'format': random.choice(['mp4', 'mp3', 'jpg', 'png']),
                    'resolution': random.choice(['1080p', '720p', '480p']) if analysis_type in ['deepfake'] else None
                },
                ip_address=f"192.168.1.{random.randint(100, 200)}",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            logs_created += 1
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {logs_created} sample detection logs')
        )
