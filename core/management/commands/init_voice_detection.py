"""
Management command to initialize and test the Voice Clone Detection System
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
import os
import tempfile
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Initialize and test the Voice Clone Detection System'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--test-audio',
            type=str,
            help='Path to test audio file'
        )
        parser.add_argument(
            '--create-sample',
            action='store_true',
            help='Create a sample audio file for testing'
        )
        parser.add_argument(
            '--health-check',
            action='store_true',
            help='Run system health check only'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Voice Clone Detection System Initialization')
        )
        
        try:
            # Import the voice detection system
            from core.voice_clone_detection import VoiceCloneDetector
            from core.voice_utils import health_monitor, AudioFileHandler
            
            # Initialize the detector
            self.stdout.write('📊 Initializing AI models...')
            detector = VoiceCloneDetector()
            init_result = detector.initialize()
            
            # Display initialization results
            self.stdout.write(f"✅ ECAPA Model Loaded: {init_result['ecapa_model_loaded']}")
            self.stdout.write(f"✅ Classifier Trained: {init_result['classifier_trained']}")
            self.stdout.write(f"✅ System Ready: {init_result['system_ready']}")
            
            if options['health_check']:
                # Run health check only
                health_report = health_monitor.get_health_report()
                self.stdout.write('\n📈 System Health Report:')
                self.stdout.write(f"Overall Health: {health_report['overall_health'].upper()}")
                self.stdout.write(f"Success Rate: {health_report['success_rate']:.1f}%")
                self.stdout.write(f"Total Analyses: {health_report['total_analyses']}")
                self.stdout.write(f"Average Processing Time: {health_report['average_processing_time']:.2f}s")
                return
            
            # Create sample audio if requested
            test_file = None
            if options['create_sample']:
                test_file = self.create_sample_audio()
                self.stdout.write(f"📄 Created sample audio: {test_file}")
            elif options['test_audio'] and os.path.exists(options['test_audio']):
                test_file = options['test_audio']
                self.stdout.write(f"📄 Using provided audio: {test_file}")
            
            # Test the system if we have a test file
            if test_file:
                self.stdout.write('\n🔍 Testing voice analysis...')
                
                # Validate the test file
                is_valid, message = AudioFileHandler.validate_audio_file(test_file)
                if not is_valid:
                    self.stdout.write(
                        self.style.ERROR(f"❌ Test file validation failed: {message}")
                    )
                    return
                
                # Run analysis
                start_time = timezone.now()
                results = detector.analyze_voice(test_file)
                end_time = timezone.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Display results
                if 'error' not in results:
                    classification = results['classification']
                    language = results['language_analysis']
                    
                    self.stdout.write('\n📊 Analysis Results:')
                    self.stdout.write(f"🎯 Result: {classification['result'].upper()}")
                    self.stdout.write(f"📈 Confidence: {classification['confidence_score']:.1f}%")
                    self.stdout.write(f"🗣️  Language: {language['language_name']} ({language['confidence']:.1f}%)")
                    self.stdout.write(f"⏱️  Processing Time: {processing_time:.2f}s")
                    
                    # Feature analysis
                    features = results.get('feature_analysis', {})
                    if features:
                        self.stdout.write('\n🔧 Feature Analysis:')
                        self.stdout.write(f"MFCC Features: {features.get('mfcc_features_count', 0)}")
                        self.stdout.write(f"Log-Mel Features: {features.get('log_mel_features_count', 0)}")
                        self.stdout.write(f"CQCC Features: {features.get('cqcc_features_count', 0)}")
                        self.stdout.write(f"ECAPA Embeddings: {features.get('ecapa_embeddings_dim', 0)}")
                        self.stdout.write(f"Total Features: {features.get('total_features', 0)}")
                    
                    # Model scores
                    model_scores = results.get('model_scores', {})
                    if model_scores:
                        self.stdout.write('\n🤖 Model Scores:')
                        self.stdout.write(f"SVM Fake Probability: {model_scores.get('svm_fake_probability', 0):.1f}%")
                        self.stdout.write(f"MLP Fake Probability: {model_scores.get('mlp_fake_probability', 0):.1f}%")
                    
                    self.stdout.write(
                        self.style.SUCCESS('\n✅ Voice Clone Detection System test completed successfully!')
                    )
                    
                else:
                    self.stdout.write(
                        self.style.ERROR(f"❌ Analysis failed: {results['error']}")
                    )
                
                # Clean up temporary file
                if options['create_sample'] and os.path.exists(test_file):
                    os.unlink(test_file)
            
            else:
                self.stdout.write(
                    self.style.WARNING('⚠️  No test audio provided. Use --create-sample or --test-audio')
                )
            
            # Display system information
            self.stdout.write('\n📋 System Information:')
            self.stdout.write(f"Model Version: 3.0")
            self.stdout.write(f"Tech Stack: ECAPA-TDNN + LibROSA + PyTorch + Scikit-learn")
            self.stdout.write(f"Supported Formats: WAV, MP3, M4A, FLAC, OGG, AAC")
            self.stdout.write(f"Max File Size: 50MB")
            self.stdout.write(f"Max Duration: 5 minutes")
            
        except ImportError as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Import error: {e}")
            )
            self.stdout.write(
                self.style.WARNING('💡 Make sure all required packages are installed: librosa, torch, scikit-learn, speechbrain')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ System initialization failed: {e}")
            )
    
    def create_sample_audio(self):
        """
        Create a sample audio file for testing
        """
        try:
            import librosa
            
            # Generate a simple test audio (3 seconds of sine wave)
            sample_rate = 16000
            duration = 3.0
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.05, len(audio))
            audio = audio + noise
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            import soundfile as sf
            sf.write(temp_file.name, audio, sample_rate)
            
            return temp_file.name
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Failed to create sample audio: {e}")
            )
            return None
