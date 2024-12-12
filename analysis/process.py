import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import soundfile as sf
import argparse
import os

class SpatialAudioProcessor:
    def __init__(self, hrtf_path=None):
        """
        Initialize the spatial audio processor
        Args:
            hrtf_path: Path to HRTF database (optional)
        """
        self.hrtf_db = self._load_hrtf_database(hrtf_path) if hrtf_path else None
        
    def process_audio(self, input_file, output_file):
        """
        Process ReSpeaker v2.0 6-channel audio into binaural output.
        Microphone data is in channels 2,3,4,5 of the input.
        Args:
            input_file: Path to 6-channel input audio file
            output_file: Path to save binaural output
        """
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Try reading with scipy.io.wavfile first as it's more robust for some WAV formats
        try:
            sample_rate, audio_data = wavfile.read(input_file)
        except Exception as e:
            print(f"Warning: scipy.io.wavfile failed to read file: {e}")
            try:
                # Fallback to soundfile
                audio_data, sample_rate = sf.read(input_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read audio file {input_file}. Error: {e}\n"
                                 f"File exists: {os.path.exists(input_file)}\n"
                                 f"File size: {os.path.getsize(input_file) if os.path.exists(input_file) else 'N/A'}\n"
                                 f"File permissions: {oct(os.stat(input_file).st_mode)[-3:] if os.path.exists(input_file) else 'N/A'}")
        
        if len(audio_data.shape) != 2 or audio_data.shape[1] != 6:
            raise ValueError("Input must be 6-channel audio")
            
        # Extract just the microphone channels (2,3,4,5)
        mic_data = audio_data[:, 2:6]
            
        # ReSpeaker v2.0 microphone array geometry (circular arrangement, radius=32mm)
        radius = 0.032  # 32mm radius
        mic_positions = np.array([
            [radius * np.cos(0), radius * np.sin(0), 0],       # Mic 0 (front)
            [radius * np.cos(np.pi/2), radius * np.sin(np.pi/2), 0],   # Mic 1 (right)
            [radius * np.cos(np.pi), radius * np.sin(np.pi), 0],     # Mic 2 (back)
            [radius * np.cos(3*np.pi/2), radius * np.sin(3*np.pi/2), 0] # Mic 3 (left)
        ])
            
        # Process each channel
        left_out = np.zeros(len(mic_data))
        right_out = np.zeros(len(mic_data))
        
        for ch in range(4):  # Process the 4 mic channels
            # Calculate spatial parameters for this channel
            azimuth, elevation, distance = self._calculate_spatial_params(
                mic_positions[ch])
            
            # Get HRTF filters for this position
            hrtf_l, hrtf_r = self._get_hrtf_filters(azimuth, elevation, sample_rate)
            
            # Apply distance attenuation
            attenuation = 1.0 / max(distance, 0.1)  # Prevent division by zero
            channel_data = mic_data[:, ch] * attenuation
            
            # Convolve with HRTF filters
            left_out += fftconvolve(channel_data, hrtf_l, mode='same')
            right_out += fftconvolve(channel_data, hrtf_r, mode='same')
            
        # Normalize output
        max_amplitude = max(np.max(np.abs(left_out)), np.max(np.abs(right_out)))
        if max_amplitude > 0:
            left_out /= max_amplitude
            right_out /= max_amplitude
            
        # Stack channels and save
        output_data = np.stack([left_out, right_out], axis=1)
        sf.write(output_file, output_data, sample_rate)
        print(f"Successfully processed {input_file}")
        print(f"Binaural output saved to {output_file}")
        
    def _calculate_spatial_params(self, position):
        """Calculate azimuth, elevation and distance from position"""
        x, y, z = position
        distance = np.sqrt(x*x + y*y + z*z)
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(x*x + y*y))
        return azimuth, elevation, distance
        
    def _get_hrtf_filters(self, azimuth, elevation, sample_rate):
        """
        Get HRTF filters for given position
        Returns simplified placeholder filters if no HRTF database loaded
        """
        if self.hrtf_db is not None:
            # Look up closest HRTF filters in database
            return self._lookup_hrtf(azimuth, elevation)
        else:
            # Return simplified placeholder filters
            # In practice, you should use real HRTF measurements
            filter_length = 128
            t = np.linspace(0, 1, filter_length)
            
            # Simple ITD and ILD simulation
            itd = 0.001 * np.sin(azimuth)  # Max 1ms ITD
            ild = np.cos(azimuth)
            
            hrtf_l = np.exp(-t) * (1 + ild)
            hrtf_r = np.exp(-t) * (1 - ild)
            
            # Apply time delay
            delay_samps = int(itd * sample_rate)
            if delay_samps > 0:
                hrtf_r = np.pad(hrtf_r, (delay_samps, 0))[:-delay_samps]
            elif delay_samps < 0:
                hrtf_l = np.pad(hrtf_l, (-delay_samps, 0))[:delay_samps]
                
            return hrtf_l, hrtf_r
            
    def _load_hrtf_database(self, path):
        """Load HRTF database from file"""
        # Implementation depends on your HRTF database format
        pass
        
    def _lookup_hrtf(self, azimuth, elevation):
        """Look up closest HRTF filters in database"""
        # Implementation depends on your HRTF database format
        pass

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process ReSpeaker v2.0 6-channel audio into binaural output')
    parser.add_argument('input_file', help='Path to input 6-channel WAV file')
    parser.add_argument('output_file', help='Path to save binaural output WAV file')
    parser.add_argument('--hrtf', help='Optional path to HRTF database', default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create processor and process audio
    try:
        processor = SpatialAudioProcessor(hrtf_path=args.hrtf)
        processor.process_audio(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error processing audio: {e}")
        exit(1)

if __name__ == "__main__":
    main()