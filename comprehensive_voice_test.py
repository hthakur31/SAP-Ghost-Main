#!/usr/bin/env python3
"""
Comprehensive Voice Clone Detection Test
Tests the system with different types of synthetic audio to demonstrate capabilities
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
import json
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.voice_clone_detection_production import VoiceCloneDetectionProduction

class ComprehensiveVoiceTester:
    """Test voice clone detection with various audio samples"""
    
    def __init__(self):
        self.detector = VoiceCloneDetectionProduction()
        self.test_results = []
        print("🧪 Comprehensive Voice Clone Detection Tester")
        print("=" * 60)
    
    def generate_real_voice_sample(self, duration: float = 3.0) -> np.ndarray:
        """Generate a realistic human voice sample"""
        sr = 22050
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        # Human voice characteristics
        base_freq = np.random.uniform(120, 200)  # Typical human range
        
        # Natural harmonic series with realistic weights
        harmonics = [1, 2, 3, 4, 5, 6]
        harmonic_weights = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1]
        
        # Natural vibrato (4-6 Hz)
        vibrato_rate = np.random.uniform(4.5, 5.5)
        vibrato_depth = 0.04
        
        signal = np.zeros(samples)
        
        for harmonic, weight in zip(harmonics, harmonic_weights):
            freq = base_freq * harmonic
            
            # Add natural vibrato
            vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            instantaneous_freq = freq * (1 + vibrato)
            
            # Generate harmonic with natural variations
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
            harmonic_signal = weight * np.sin(phase)
            
            # Add natural amplitude modulation
            amplitude_variation = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
            harmonic_signal *= amplitude_variation
            
            signal += harmonic_signal
        
        # Natural envelope (attack, sustain, decay)
        envelope = np.ones(samples)
        attack_samples = int(0.1 * sr)
        decay_samples = int(0.2 * sr)
        
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        signal *= envelope
        
        # Add natural breath noise
        breath_noise = 0.001 * np.random.randn(samples)
        signal += breath_noise
        
        # Normalize
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.7
        
        return signal.astype(np.float32)
    
    def generate_fake_voice_sample(self, duration: float = 3.0) -> np.ndarray:
        """Generate a fake/cloned voice sample with AI artifacts"""
        sr = 22050
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        # AI voice characteristics (less natural)
        base_freq = np.random.uniform(130, 180)
        
        # More uniform harmonic distribution (AI artifact)
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
        harmonic_weights = [1.0, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15]  # Too uniform
        
        # Slightly irregular vibrato (AI artifact)
        vibrato_rate = np.random.uniform(3, 7)  # More variable
        vibrato_depth = 0.08  # More pronounced
        
        signal = np.zeros(samples)
        
        for harmonic, weight in zip(harmonics, harmonic_weights):
            freq = base_freq * harmonic
            
            # Add artificial vibrato patterns
            vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t + np.random.uniform(0, 2*np.pi))
            instantaneous_freq = freq * (1 + vibrato)
            
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
            harmonic_signal = weight * np.sin(phase)
            
            signal += harmonic_signal
        
        # More mechanical envelope (AI artifact)
        envelope = np.ones(samples)
        attack_samples = int(0.05 * sr)  # Too fast attack
        decay_samples = int(0.1 * sr)   # Too fast decay
        
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        signal *= envelope
        
        # Add digital artifacts
        # Quantization noise (AI artifact)
        quantization_noise = 0.002 * np.random.randint(-2, 3, samples)
        
        # Slight periodic artifacts (AI processing artifacts)
        artifact_freq = sr / 256  # Typical AI processing artifact
        processing_artifact = 0.001 * np.sin(2 * np.pi * artifact_freq * t)
        
        signal += quantization_noise + processing_artifact
        
        # Normalize
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.7
        
        return signal.astype(np.float32)
    
    def test_audio_sample(self, audio: np.ndarray, expected_label: str, description: str):
        """Test a single audio sample"""
        print(f"\n🔍 Testing: {description}")
        print(f"   Expected: {expected_label.upper()}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, 22050)
            
            try:
                # Analyze with detector
                result = self.detector.predict_file(tmp_file.name)
                
                if result and result.get('success', False) and 'classification' in result:
                    classification = result['classification']
                    prediction = classification['result']
                    confidence = classification['confidence']
                    
                    # Determine if prediction is correct
                    is_correct = (
                        (expected_label.lower() == 'real' and prediction.lower() == 'real') or
                        (expected_label.lower() == 'fake' and prediction.lower() == 'fake')
                    )
                    
                    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    print(f"   Result: {prediction.upper()}")
                    print(f"   Confidence: {confidence:.1f}%")
                    print(f"   Status: {status}")
                    
                    # Store result
                    self.test_results.append({
                        'description': description,
                        'expected': expected_label,
                        'predicted': prediction,
                        'confidence': confidence,
                        'correct': is_correct,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                else:
                    print("   ❌ Analysis failed")
                    self.test_results.append({
                        'description': description,
                        'expected': expected_label,
                        'predicted': 'ERROR',
                        'confidence': 0.0,
                        'correct': False,
                        'timestamp': datetime.now().isoformat()
                    })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
    
    def run_comprehensive_tests(self):
        """Run a comprehensive set of tests"""
        print("🚀 Starting comprehensive voice clone detection tests...")
        
        # Initialize detector
        if not self.detector.initialize():
            print("❌ Failed to initialize detector!")
            return False
        
        print("\n📊 Running test battery...")
        
        # Test 1: Natural human voice samples
        print("\n" + "="*50)
        print("TEST CATEGORY 1: NATURAL HUMAN VOICES")
        print("="*50)
        
        for i in range(3):
            audio = self.generate_real_voice_sample(duration=3.0)
            self.test_audio_sample(audio, 'real', f'Natural human voice #{i+1}')
        
        # Test 2: AI/Cloned voice samples
        print("\n" + "="*50)
        print("TEST CATEGORY 2: AI/CLONED VOICES")
        print("="*50)
        
        for i in range(3):
            audio = self.generate_fake_voice_sample(duration=3.0)
            self.test_audio_sample(audio, 'fake', f'AI/Cloned voice #{i+1}')
        
        # Test 3: Edge cases
        print("\n" + "="*50)
        print("TEST CATEGORY 3: EDGE CASES")
        print("="*50)
        
        # Very short real voice
        short_real = self.generate_real_voice_sample(duration=1.0)
        self.test_audio_sample(short_real, 'real', 'Short natural voice (1s)')
        
        # Very short fake voice
        short_fake = self.generate_fake_voice_sample(duration=1.0)
        self.test_audio_sample(short_fake, 'fake', 'Short AI voice (1s)')
        
        # Longer samples
        long_real = self.generate_real_voice_sample(duration=6.0)
        self.test_audio_sample(long_real, 'real', 'Long natural voice (6s)')
        
        long_fake = self.generate_fake_voice_sample(duration=6.0)
        self.test_audio_sample(long_fake, 'fake', 'Long AI voice (6s)')
        
        # Generate final report
        self.generate_report()
        
        return True
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        correct_predictions = sum(1 for r in self.test_results if r['correct'])
        accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.1f}%")
        
        # Category breakdown
        real_tests = [r for r in self.test_results if r['expected'].lower() == 'real']
        fake_tests = [r for r in self.test_results if r['expected'].lower() == 'fake']
        
        real_correct = sum(1 for r in real_tests if r['correct'])
        fake_correct = sum(1 for r in fake_tests if r['correct'])
        
        print(f"\n📊 Category Performance:")
        print(f"Real Voice Detection: {real_correct}/{len(real_tests)} ({(real_correct/len(real_tests))*100:.1f}%)")
        print(f"Fake Voice Detection: {fake_correct}/{len(fake_tests)} ({(fake_correct/len(fake_tests))*100:.1f}%)")
        
        # Confidence analysis
        confidences = [r['confidence'] for r in self.test_results if r['correct']]
        if confidences:
            avg_confidence = np.mean(confidences)
            print(f"\n🎯 Average Confidence (Correct Predictions): {avg_confidence:.1f}%")
        
        # Detailed results
        print(f"\n📝 Detailed Results:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result['correct'] else "❌"
            print(f"{i:2d}. {status_icon} {result['description']}")
            print(f"     Expected: {result['expected'].upper()}, Got: {result['predicted'].upper()}, Confidence: {result['confidence']:.1f}%")
        
        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'real_accuracy': (real_correct/len(real_tests))*100 if real_tests else 0,
                    'fake_accuracy': (fake_correct/len(fake_tests))*100 if fake_tests else 0,
                    'avg_confidence': np.mean(confidences) if confidences else 0
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
        # Final assessment
        if accuracy >= 80:
            print(f"\n🎉 EXCELLENT! Your voice clone detection system is performing very well!")
            print(f"   The system successfully detected {accuracy:.1f}% of voice samples correctly.")
        elif accuracy >= 60:
            print(f"\n👍 GOOD! Your system shows promising results with {accuracy:.1f}% accuracy.")
            print(f"   Consider further training or parameter tuning for improvement.")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT. {accuracy:.1f}% accuracy suggests the system needs more training data.")
            print(f"   Consider retraining with more diverse samples.")

def main():
    """Run comprehensive voice clone detection tests"""
    try:
        tester = ComprehensiveVoiceTester()
        success = tester.run_comprehensive_tests()
        
        if success:
            print(f"\n🏁 Testing completed successfully!")
        else:
            print(f"\n❌ Testing failed!")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
