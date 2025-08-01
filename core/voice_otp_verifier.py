"""
Voice OTP Verification System
Handles speech-to-text conversion and OTP verification
"""

import speech_recognition as sr
import logging
import random
import string
import io
import tempfile
import os
from typing import Dict, Any, Optional
from pydub import AudioSegment
from pydub.utils import which

logger = logging.getLogger(__name__)

class VoiceOTPVerifier:
    """Handles Voice OTP generation and verification"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
    def generate_otp(self, length: int = 6) -> str:
        """Generate a random OTP"""
        digits = string.digits
        otp = ''.join(random.choice(digits) for _ in range(length))
        logger.info(f"Generated OTP: {otp}")
        return otp
    
    def record_and_recognize_speech_from_file(self, audio_file) -> Dict[str, Any]:
        """
        Convert uploaded audio file to text
        Returns: Dict with success status and recognized text or error
        """
        temp_file_path = None
        wav_file_path = None
        
        try:
            # Create a temporary file to store the original audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                # Write the uploaded file content to temporary file
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            logger.info(f"🎵 Processing audio file: {audio_file.name} (size: {audio_file.size} bytes)")
            
            # Try multiple approaches to convert the audio
            wav_file_path = None
            
            # Approach 1: Try pydub conversion with better error handling
            try:
                logger.info("🔄 Attempting pydub conversion...")
                
                # Try to load the audio file (pydub can handle many formats)
                audio = AudioSegment.from_file(temp_file_path)
                
                # Convert to WAV format with specific parameters
                wav_file_path = temp_file_path.replace('.webm', '.wav')
                audio.export(
                    wav_file_path, 
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
                )
                
                logger.info(f"✅ Successfully converted audio to WAV format")
                
            except Exception as pydub_error:
                logger.warning(f"⚠ Pydub conversion failed: {pydub_error}")
                
                # Approach 2: Try to use the original file directly
                try:
                    logger.info("🔄 Trying to use original file directly...")
                    wav_file_path = temp_file_path
                    
                    # Test if the file can be read by speech_recognition
                    with sr.AudioFile(wav_file_path) as test_source:
                        self.recognizer.record(test_source, duration=0.1)
                    
                    logger.info("✅ Original file format is compatible")
                    
                except Exception as direct_error:
                    logger.warning(f"⚠ Direct file access failed: {direct_error}")
                    
                    # Approach 3: Create a simple WAV file manually (fallback)
                    logger.info("🔄 Creating fallback WAV file...")
                    wav_file_path = temp_file_path.replace('.webm', '.wav')
                    
                    # For demo purposes, if all else fails, use simulation
                    raise Exception("Audio format conversion failed - falling back to simulation")
            
            # Read the WAV file with speech_recognition
            with sr.AudioFile(wav_file_path) as source:
                logger.info("🎙 Processing converted audio file...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
                
                logger.info("🔄 Converting speech to text...")
                # Convert audio to text using Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
                
                logger.info(f"📝 Recognized text: '{text}'")
                
                return {
                    "success": True,
                    "text": text,
                    "message": "Speech recognized successfully from uploaded file"
                }
        
        except sr.UnknownValueError:
            logger.warning("❌ Could not understand audio from file")
            return {
                "success": False,
                "text": "",
                "error": "UNKNOWN_VALUE",
                "message": "Could not understand the audio. Please speak clearly and try again."
            }
            
        except sr.RequestError as e:
            logger.error(f"⚠ Speech recognition service error: {e}")
            return {
                "success": False,
                "text": "",
                "error": "REQUEST_ERROR",
                "message": "Speech recognition service unavailable. Check internet connection."
            }
        
        except Exception as e:
            logger.error(f"🚨 Audio processing failed, will use simulation: {e}")
            # Return a special error that triggers simulation in the view
            return {
                "success": False,
                "text": "",
                "error": "CONVERSION_FAILED",
                "message": "Audio format conversion failed - using simulation for demo"
            }
        
        finally:
            # Clean up temporary files
            for file_path in [temp_file_path, wav_file_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        logger.info(f"🧹 Cleaned up temporary file: {os.path.basename(file_path)}")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠ Could not clean up temporary file: {cleanup_error}")

    def record_and_recognize_speech(self, timeout: int = 5, phrase_timeout: int = 2) -> Dict[str, Any]:
        """
        Record audio from microphone and convert to text
        Returns: Dict with success status and recognized text or error
        """
        try:
            with sr.Microphone() as source:
                logger.info("🎙 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("🎙 Listening for speech...")
                # Listen with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_timeout
                )
                
                logger.info("🔄 Converting speech to text...")
                # Convert audio to text using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                logger.info(f"📝 Recognized text: '{text}'")
                
                return {
                    "success": True,
                    "text": text,
                    "message": "Speech recognized successfully"
                }
                
        except sr.WaitTimeoutError:
            logger.warning("⏰ Listening timeout - no speech detected")
            return {
                "success": False,
                "text": "",
                "error": "TIMEOUT",
                "message": "No speech detected. Please try again."
            }
            
        except sr.UnknownValueError:
            logger.warning("❌ Could not understand audio")
            return {
                "success": False,
                "text": "",
                "error": "UNKNOWN_VALUE",
                "message": "Could not understand the audio. Please speak clearly."
            }
            
        except sr.RequestError as e:
            logger.error(f"⚠ Speech recognition service error: {e}")
            return {
                "success": False,
                "text": "",
                "error": "REQUEST_ERROR",
                "message": "Speech recognition service unavailable. Check internet connection."
            }
            
        except Exception as e:
            logger.error(f"🚨 Unexpected error in speech recognition: {e}")
            return {
                "success": False,
                "text": "",
                "error": "GENERAL_ERROR",
                "message": f"An unexpected error occurred: {str(e)}"
            }
    
    def extract_digits_from_text(self, text: str) -> str:
        """Extract only digits from recognized text"""
        digits = ''.join(filter(str.isdigit, text))
        logger.info(f"📊 Extracted digits from '{text}': '{digits}'")
        return digits
    
    def verify_voice_otp(self, original_otp: str, spoken_text: str) -> Dict[str, Any]:
        """
        Verify if spoken OTP matches the original OTP
        Returns: Dict with verification result and details
        """
        try:
            # Extract digits from spoken text
            spoken_digits = self.extract_digits_from_text(spoken_text)
            
            # Compare OTPs
            is_match = spoken_digits == original_otp
            
            # Calculate confidence based on exact match
            confidence = 100.0 if is_match else 0.0
            
            # If partial match, calculate partial confidence
            if not is_match and len(spoken_digits) == len(original_otp):
                matches = sum(1 for a, b in zip(spoken_digits, original_otp) if a == b)
                confidence = (matches / len(original_otp)) * 100
            
            result = {
                "success": True,
                "is_verified": is_match,
                "original_otp": original_otp,
                "spoken_text": spoken_text,
                "extracted_digits": spoken_digits,
                "confidence": confidence,
                "message": "✅ OTP verified successfully!" if is_match else "❌ OTP verification failed"
            }
            
            logger.info(f"🔍 OTP Verification: {result}")
            return result
            
        except Exception as e:
            logger.error(f"🚨 Error in OTP verification: {e}")
            return {
                "success": False,
                "is_verified": False,
                "error": str(e),
                "message": "An error occurred during verification"
            }
    
    def full_voice_otp_verification_from_file(self, original_otp: str, audio_file) -> Dict[str, Any]:
        """
        Complete voice OTP verification process from uploaded audio file:
        1. Process uploaded audio file
        2. Convert to text  
        3. Verify against original OTP
        """
        logger.info(f"🎯 Starting full voice OTP verification from file for OTP: {original_otp}")
        
        # Step 1: Process uploaded audio file and recognize speech
        speech_result = self.record_and_recognize_speech_from_file(audio_file)
        
        if not speech_result["success"]:
            return {
                "success": False,
                "is_verified": False,
                "stage": "SPEECH_RECOGNITION",
                "error": speech_result.get("error", "UNKNOWN"),
                "message": speech_result.get("message", "Speech recognition failed")
            }
        
        # Step 2: Verify OTP
        verification_result = self.verify_voice_otp(original_otp, speech_result["text"])
        
        # Combine results
        final_result = {
            "success": True,
            "is_verified": verification_result["is_verified"],
            "original_otp": original_otp,
            "spoken_text": speech_result["text"],
            "extracted_digits": verification_result["extracted_digits"],
            "confidence": verification_result["confidence"],
            "stage": "COMPLETE",
            "message": verification_result["message"]
        }
        
        logger.info(f"🏁 Final verification result from file: {final_result}")
        return final_result

    def full_voice_otp_verification(self, original_otp: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Complete voice OTP verification process:
        1. Record speech
        2. Convert to text  
        3. Verify against original OTP
        """
        logger.info(f"🎯 Starting full voice OTP verification for OTP: {original_otp}")
        
        # Step 1: Record and recognize speech
        speech_result = self.record_and_recognize_speech(timeout=timeout)
        
        if not speech_result["success"]:
            return {
                "success": False,
                "is_verified": False,
                "stage": "SPEECH_RECOGNITION",
                "error": speech_result.get("error", "UNKNOWN"),
                "message": speech_result.get("message", "Speech recognition failed")
            }
        
        # Step 2: Verify OTP
        verification_result = self.verify_voice_otp(original_otp, speech_result["text"])
        
        # Combine results
        final_result = {
            "success": True,
            "is_verified": verification_result["is_verified"],
            "original_otp": original_otp,
            "spoken_text": speech_result["text"],
            "extracted_digits": verification_result["extracted_digits"],
            "confidence": verification_result["confidence"],
            "stage": "COMPLETE",
            "message": verification_result["message"]
        }
        
        logger.info(f"🏁 Final verification result: {final_result}")
        return final_result

# Global instance
voice_otp_verifier = VoiceOTPVerifier()
