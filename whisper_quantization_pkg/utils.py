import torch
import psutil
import os
from pathlib import Path
import time
import pandas as pd
# Remove numpy since it's not used
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer
from tqdm.notebook import tqdm  # Add tqdm import

class ModelProfiler:
    """Class to handle model profiling and metrics"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_history = []
        self.size_details = {}  # Store detailed size information
    
    def measure_model_size(self, model: torch.nn.Module) -> float:
        """Measure model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def get_detailed_model_size(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get detailed size information about the model components"""
        param_size = 0
        buffer_size = 0
        layer_info = {}
        
        # Parameters
        total_params = 0
        for name, param in model.named_parameters():
            num_params = param.nelement()
            total_params += num_params
            size = num_params * param.element_size()
            param_size += size
            layer_info[name] = {
                'size_mb': size / 1024**2,
                'num_params': num_params,
                'type': 'parameter'
            }
        
        # Buffers
        for name, buffer in model.named_buffers():
            size = buffer.nelement() * buffer.element_size()
            buffer_size += size
            layer_info[name] = {
                'size_mb': size / 1024**2,
                'num_elements': buffer.nelement(),
                'type': 'buffer'
            }
        
        total_size = (param_size + buffer_size) / 1024**2
        
        self.size_details = {
            'total_size_mb': total_size,
            'param_size_mb': param_size / 1024**2,
            'buffer_size_mb': buffer_size / 1024**2,
            'total_params': total_params,
            'layer_info': layer_info
        }
        
        return self.size_details
    
    def print_size_analysis(self):
        """Print detailed size analysis"""
        if not self.size_details:
            print("No size analysis available. Run get_detailed_model_size first.")
            return
        
        print(f"Model Size Analysis for {self.model_name}")
        print("-" * 50)
        print(f"Total Model Size: {self.size_details['total_size_mb']:.2f} MB")
        print(f"Parameter Size: {self.size_details['param_size_mb']:.2f} MB")
        print(f"Buffer Size: {self.size_details['buffer_size_mb']:.2f} MB")
        print(f"Total Parameters: {self.size_details['total_params']:,}")
        print("\nLayer-by-Layer Breakdown:")
        print("-" * 50)
        
        # Sort layers by size
        sorted_layers = sorted(
            self.size_details['layer_info'].items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )
        
        for name, info in sorted_layers:
            if info['type'] == 'parameter':
                print(f"{name}:")
                print(f"  Size: {info['size_mb']:.2f} MB")
                print(f"  Parameters: {info['num_params']:,}")
            else:
                print(f"Buffer {name}:")
                print(f"  Size: {info['size_mb']:.2f} MB")
            print("-" * 30)

class VisualizationUtils:
    """Class for visualization utilities"""
    
    @staticmethod
    def set_style():
        """Set default style for plots"""
        sns.set_theme()  # Use seaborn's default theme
        plt.style.use('seaborn-v0_8')  # Use seaborn's style in matplotlib
    
    @staticmethod
    def plot_error_distributions(results_df: pd.DataFrame, save_path: str = None):
        """Plot WER and CER distributions"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=results_df, x='wer', bins=20)
        plt.title('Distribution of Word Error Rate')
        plt.xlabel('WER')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=results_df, x='cer', bins=20)
        plt.title('Distribution of Character Error Rate')
        plt.xlabel('CER')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_performance_metrics(results_df: pd.DataFrame, save_path: str = None):
        """Plot performance metrics distributions"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=results_df, x='inference_time', bins=20)
        plt.title('Distribution of Inference Time')
        plt.xlabel('Time (seconds)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=results_df, x='memory_used', bins=20)
        plt.title('Distribution of Memory Usage')
        plt.xlabel('Memory (MB)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def setup_device() -> torch.device:
    """Configure the device (MPS for M3, CPU otherwise)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
        # Fix f-string issue by adding proper placeholder
        print(f"PyTorch MPS device properties: {torch.backends.mps.is_built()}")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    return device

def ensure_dirs_exist():
    """Ensure all necessary directories exist"""
    dirs = ['data', 'models', 'results', 'results/plots']
    for d in dirs:
        Path(d).mkdir(exist_ok=True, parents=True)

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"

class TranscriptionMetrics:
    """Class to handle transcription-specific metrics"""
    
    @staticmethod
    def print_sample_comparisons(results_df: pd.DataFrame, n_samples: int = 3):
        """Print sample transcription comparisons"""
        print("Sample Transcriptions:")
        for _, row in results_df.head(n_samples).iterrows():
            print("\nReference:")
            print(row['reference'])
            print("\nHypothesis:")
            print(row['hypothesis'])
            print(f"WER: {row['wer']:.4f}, CER: {row['cer']:.4f}")
            print("-" * 80)
    
    @staticmethod
    def calculate_summary_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary metrics for transcription results"""
        return {
            'avg_wer': results_df['wer'].mean(),
            'median_wer': results_df['wer'].median(),
            'std_wer': results_df['wer'].std(),
            'avg_cer': results_df['cer'].mean(),
            'median_cer': results_df['cer'].median(),
            'std_cer': results_df['cer'].std(),
            'avg_inference_time': results_df['inference_time'].mean(),
            'avg_memory_used': results_df['memory_used'].mean(),
        }
        
# Add this at the end of your current utils.py

class WhisperEvaluator:
    """Class to handle Whisper model evaluation"""
    
    def __init__(self, model, processor, device, profiler: ModelProfiler):
        self.model = model
        self.processor = processor
        self.device = device
        self.profiler = profiler
        self.model.to(device)
        self.model.eval()
    
    def process_audio(self, audio):
        """Process audio input for model"""
        return self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
    
    def evaluate_sample(self, audio, reference):
        """Evaluate a single audio sample"""
        input_features = self.process_audio(audio)
        
        # Measure inference
        mem_before = self.profiler.get_memory_usage()
        start_time = time.time()
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
        
        # Calculate metrics
        inference_time = time.time() - start_time
        mem_used = self.profiler.get_memory_usage() - mem_before
        wer, cer = self.profiler.calculate_error_metrics(reference, transcription)
        
        return {
            'reference': reference,
            'hypothesis': transcription,
            'wer': wer,
            'cer': cer,
            'inference_time': inference_time,
            'memory_used': mem_used
        }
    
    def evaluate_dataset(self, dataset):
        """Evaluate entire dataset"""
        results = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            result = self.evaluate_sample(
                item["audio"]["array"],
                item["text"]
            )
            result['sample_id'] = idx
            results.append(result)
            
            # Log metrics
            self.profiler.log_metrics({
                'wer': result['wer'],
                'cer': result['cer'],
                'inference_time': result['inference_time'],
                'memory_used': result['memory_used']
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, path: str):
        """Save model and processor"""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"Model and processor saved to '{path}'")
        
