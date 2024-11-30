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
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    
    @staticmethod
    def calculate_error_metrics(reference: str, hypothesis: str) -> Tuple[float, float]:
        """Calculate WER and CER"""
        return wer(reference, hypothesis), cer(reference, hypothesis)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics with timestamp"""
        metrics['timestamp'] = time.time()
        metrics['model_name'] = self.model_name
        self.metrics_history.append(metrics)
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as DataFrame"""
        return pd.DataFrame(self.metrics_history)
    
    def save_metrics(self, filepath: str):
        """Save metrics to CSV"""
        self.get_metrics_df().to_csv(filepath, index=False)

class VisualizationUtils:
    """Class for visualization utilities"""
    
    @staticmethod
    def set_style():
        """Set default style for plots"""
        plt.style.use('seaborn')
        sns.set_palette('deep')
    
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
        
