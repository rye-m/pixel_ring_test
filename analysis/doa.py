import numpy as np
from scipy.io import wavfile
import soundfile as sf
import argparse
import os
from scipy.signal import correlate, find_peaks
import math
import time

class DOADetector:
    def __init__(self):
        """Initialize the DOA detector for ReSpeaker v2.0"""
        # ReSpeaker v2.0 parameters
        self.radius = 0.032  # Array radius in meters
        self.sound_speed = 343.0  # Speed of sound in m/s
        self.sample_rate = None
        
    def process_audio(self, input_file, window_size=4096, overlap=0.5):
        """
        Process 6-channel audio to find direction of loudest sound
        Args:
            input_file: Path to 6-channel input audio file
            window_size: Size of processing window in samples
            overlap: Overlap between windows (0-1)
        """
        # Read audio file
        try:
            audio_data, self.sample_rate = sf.read(input_file)
        except Exception as e:
            try:
                self.sample_rate, audio_data = wavfile.read(input_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read audio file: {e}")
                
        if len(audio_data.shape) != 2 or audio_data.shape[1] != 6:
            raise ValueError("Input must be 6-channel audio")
            
        # Extract microphone channels (2,3,4,5 for ReSpeaker v2.0)
        mic_data = audio_data[:, 2:6]
        
        # Calculate window parameters
        hop_size = int(window_size * (1 - overlap))
        num_windows = (len(mic_data) - window_size) // hop_size + 1
        
        # Print header
        print("\nTime(s) | Angle° | Confidence | Direction")
        print("-" * 45)
        
        # Process each window
        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            window = mic_data[start:end]
            
            # Calculate timestamp for this window
            timestamp = start / self.sample_rate
            
            # Calculate energy for this window
            energy = np.sum(window ** 2)
            
            # Only process if energy is above threshold
            if energy > 1e-6:  # Adjust threshold as needed
                # Calculate DOA for this window
                angle, confidence = self._calculate_doa(window)
                
                # Get cardinal direction
                direction = self._get_direction(angle) if confidence > 0.3 else "?"
                
                # Print results
                print(f"{timestamp:6.2f} | {angle:6.1f} | {confidence:9.2f} | {direction:9}")
            
            # Optional: add small delay to simulate real-time playback
            # time.sleep(hop_size / self.sample_rate)
        
    def _calculate_doa(self, window):
        """
        Calculate direction of arrival using cross-correlation
        Returns angle in degrees and confidence measure
        """
        # Mic pairs for cross-correlation
        mic_pairs = [(0,1), (1,2), (2,3), (3,0)]  # Adjacent pairs
        
        max_tdoa = self.radius * 2 / self.sound_speed
        max_delay = int(max_tdoa * self.sample_rate)
        
        angles = []
        confidences = []
        
        for mic1, mic2 in mic_pairs:
            # Cross-correlate the two signals
            correlation = correlate(window[:, mic1], window[:, mic2], mode='full')
            mid_point = len(correlation) // 2
            
            # Find peaks in correlation
            peaks, properties = find_peaks(correlation, distance=max_delay)
            if len(peaks) == 0:
                continue
                
            # Get the strongest peak
            peak_idx = peaks[np.argmax(correlation[peaks])]
            delay = peak_idx - mid_point
            
            # Calculate angle from delay
            if abs(delay) > max_delay:
                continue
                
            # Calculate angle between mic pair
            base_angle = 90 * mic1  # 0°, 90°, 180°, 270° for mics 0,1,2,3
            pair_angle = math.degrees(math.asin((delay / self.sample_rate * self.sound_speed) / (2 * self.radius)))
            
            if mic1 in [0, 2]:  # Front-back pairs
                angle = base_angle + pair_angle
            else:  # Left-right pairs
                angle = base_angle - pair_angle
                
            # Normalize angle to 0-360
            angle = angle % 360
            
            # Calculate confidence based on peak height
            peak_height = correlation[peak_idx]
            confidence = abs(peak_height) / np.max(np.abs(correlation))
            
            angles.append(angle)
            confidences.append(confidence)
            
        if not angles:
            return 0, 0
            
        # Weight angles by their confidences
        weighted_angle = np.average(angles, weights=confidences)
        avg_confidence = np.mean(confidences)
        
        return weighted_angle, avg_confidence

    def _get_direction(self, angle):
        """Convert angle to cardinal direction"""
        if 45 <= angle < 135:
            return "East"
        elif 135 <= angle < 225:
            return "South"
        elif 225 <= angle < 315:
            return "West"
        else:
            return "North"

def main():
    parser = argparse.ArgumentParser(description='Calculate direction of arrival over time')
    parser.add_argument('input_file', help='Path to input 6-channel WAV file')
    parser.add_argument('--window-size', type=int, default=4096,
                      help='Analysis window size in samples (default: 4096)')
    parser.add_argument('--overlap', type=float, default=0.75,
                      help='Window overlap factor 0-1 (default: 0.75)')
    
    args = parser.parse_args()
    
    print("\nDirection of Arrival Analysis")
    print(f"Processing file: {args.input_file}")
    print(f"Window size: {args.window_size} samples")
    print(f"Overlap: {args.overlap * 100}%")
    
    print("\nReference:")
    print("  0° = North (front)")
    print(" 90° = East (right)")
    print("180° = South (back)")
    print("270° = West (left)")
    
    try:
        detector = DOADetector()
        detector.process_audio(
            args.input_file, 
            window_size=args.window_size,
            overlap=args.overlap
        )
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        exit(1)

if __name__ == "__main__":
    main()