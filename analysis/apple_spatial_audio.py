import numpy as np
from scipy.io import wavfile
import soundfile as sf
import argparse
import os
from scipy.signal import fftconvolve
import wave
import struct

class AppleSpatialProcessor:
    def __init__(self):
        """Initialize the spatial audio processor for Apple device compatibility"""
        self.sample_rate = None
        
    def process_audio(self, input_file, output_file):
        """
        Process 6-channel ReSpeaker audio into Apple-compatible spatial audio
        Args:
            input_file: Path to 6-channel input audio file
            output_file: Path to save spatial audio output
        """
        try:
            audio_data, self.sample_rate = sf.read(input_file)
        except Exception as e:
            try:
                self.sample_rate, audio_data = wavfile.read(input_file)
                # Convert to float if integer data
                if audio_data.dtype.kind in 'iu':
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            except Exception as e:
                raise RuntimeError(f"Failed to read audio file: {e}")
            
        if len(audio_data.shape) != 2 or audio_data.shape[1] != 6:
            raise ValueError("Input must be 6-channel audio")
            
        # Extract microphone channels (2,3,4,5 for ReSpeaker v2.0)
        mic_data = audio_data[:, 2:6]
        
        # Convert to 5.1 surround format
        surround_audio = self._convert_to_surround(mic_data)
        
        # Save as WAV with specific channel ordering for spatial audio
        self._save_spatial_wav(surround_audio, output_file)
        
    def _convert_to_surround(self, mic_data):
        """Convert 4-channel mic array data to 5.1 surround format"""
        num_samples = len(mic_data)
        surround = np.zeros((num_samples, 6))  # 5.1 format
        
        # Enhanced channel mixing for better spatial effect
        # Front channels
        surround[:, 0] = 0.7 * mic_data[:, 0] + 0.3 * mic_data[:, 3]  # Left
        surround[:, 1] = 0.7 * mic_data[:, 0] + 0.3 * mic_data[:, 1]  # Right
        
        # Center (weighted mix of front and side mics)
        surround[:, 2] = 0.8 * mic_data[:, 0] + 0.1 * (mic_data[:, 1] + mic_data[:, 3])
        
        # LFE (low frequencies from all channels)
        surround[:, 3] = self._extract_lfe(mic_data)
        
        # Surround channels (enhanced mix for better spatial separation)
        surround[:, 4] = 0.6 * mic_data[:, 2] + 0.4 * mic_data[:, 3]  # Surround left
        surround[:, 5] = 0.6 * mic_data[:, 2] + 0.4 * mic_data[:, 1]  # Surround right
        
        return surround
        
    def _extract_lfe(self, mic_data, cutoff_freq=120):
        """Extract low frequencies for LFE channel"""
        # Simple lowpass filter
        nyquist = self.sample_rate / 2
        cutoff_normalized = cutoff_freq / nyquist
        
        # Create a simple FIR lowpass filter
        filter_length = 101
        h = np.sinc(2 * cutoff_normalized * (np.arange(filter_length) - (filter_length-1) / 2))
        h *= np.hamming(filter_length)
        h /= np.sum(h)
        
        # Apply to all channels and sum
        lfe = np.zeros(len(mic_data))
        for channel in range(mic_data.shape[1]):
            lfe += np.convolve(mic_data[:, channel], h, mode='same')
            
        return lfe / mic_data.shape[1]
        
    def _save_spatial_wav(self, audio_data, output_file):
        """Save audio in a format compatible with Apple spatial audio"""
        # Ensure audio is float32 and in range [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 24-bit PCM format (which Apple devices handle well)
        audio_int = (audio_data * (2**23-1)).astype(np.int32)
        
        # Save using soundfile with specific format
        sf.write(
            output_file,
            audio_data,
            self.sample_rate,
            format='WAV',
            subtype='PCM_24'
        )

def main():
    parser = argparse.ArgumentParser(
        description='Convert ReSpeaker audio to Apple-compatible spatial audio')
    parser.add_argument('input_file', help='Path to input 6-channel WAV file')
    parser.add_argument('output_file', help='Path to save spatial audio output')
    
    args = parser.parse_args()
    
    try:
        processor = AppleSpatialProcessor()
        processor.process_audio(args.input_file, args.output_file)
        print(f"\nProcessed {args.input_file}")
        print(f"Created spatial audio file: {args.output_file}")
        print("\nTo use with Apple devices:")
        print("1. Transfer the file to your iOS device")
        print("2. Play using Apple Music or Files app")
        print("3. Use AirPods Pro/Max or other compatible headphones")
        print("4. The audio should work with head tracking, though it may")
        print("   not be as precise as native Apple spatial audio content")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        exit(1)

if __name__ == "__main__":
    main()