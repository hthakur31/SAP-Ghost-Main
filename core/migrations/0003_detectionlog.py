# Generated manually for new DetectionLog model

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0002_alter_frauddetectionlog_options_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DetectionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('request_id', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('analysis_type', models.CharField(choices=[('deepfake', 'Deepfake Detection'), ('voice', 'Voice Authentication'), ('fraud', 'Fraud Detection'), ('otp', 'OTP Verification')], max_length=20)),
                ('result', models.CharField(choices=[('verified', 'Verified'), ('suspicious', 'Suspicious'), ('authentic', 'Authentic'), ('fraudulent', 'Fraudulent'), ('rejected', 'Rejected')], max_length=20)),
                ('confidence', models.FloatField(help_text='Confidence percentage (0-100)')),
                ('subscription_plan', models.CharField(choices=[('FREEMIUM', 'Freemium'), ('PREMIUM', 'Premium'), ('ENTERPRISE', 'Enterprise')], default='FREEMIUM', max_length=20)),
                ('uploaded_file', models.FileField(blank=True, null=True, upload_to='detection_files/')),
                ('result_file', models.FileField(blank=True, null=True, upload_to='result_files/')),
                ('metadata', models.JSONField(blank=True, default=dict)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.TextField(blank=True, null=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Detection Log',
                'verbose_name_plural': 'Detection Logs',
                'ordering': ['-timestamp'],
            },
        ),
    ]
