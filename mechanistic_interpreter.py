"""
Mechanistic Interpretability Tool for Qwen2.5-0.5B
A comprehensive toolkit for analyzing internal mechanisms of language models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ActivationRecord:
    """Stores activation data with metadata"""
    layer_name: str
    activations: torch.Tensor
    tokens: List[str]
    positions: List[int]

class MechanisticInterpreter:
    """Main class for mechanistic interpretability analysis"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        """
        Initialize the interpreter with a model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for better interpretability
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.hooks = []  # Store hooks for cleanup
        self.activations = {}  # Store captured activations
        
        print(f"Model loaded on {self.device}")
        print(f"Model has {self.model.config.num_hidden_layers} layers")
        print(f"Hidden size: {self.model.config.hidden_size}")
        print(f"Attention heads: {self.model.config.num_attention_heads}")
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get a module by its name path"""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
    
    def register_hooks(self, layer_names: List[str]) -> None:
        """
        Register forward hooks to capture activations.
        
        Args:
            layer_names: List of layer names to hook (e.g., ['model.layers.0.mlp'])
        """
        self.clear_hooks()
        self.activations = {}
        
        def make_hook(name):
            def hook(module, input, output):
                # Store the output activation
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        for name in layer_names:
            module = self._get_module_by_name(name)
            hook = module.register_forward_hook(make_hook(name))
            self.hooks.append(hook)
            
        print(f"Registered hooks for {len(layer_names)} layers")
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Run text through model and capture activations.
        
        Args:
            text: Input text to analyze
            max_length: Maximum sequence length
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Run forward pass (activations captured by hooks)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Add token information to activations
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        result = {}
        for name, activations in self.activations.items():
            result[name] = {
                'activations': activations,
                'tokens': tokens,
                'logits': outputs.logits.cpu(),
                'input_ids': inputs['input_ids'].cpu()
            }
        
        return result
    
    def get_layer_names(self, layer_types: List[str] = None) -> List[str]:
        """
        Get names of all layers of specified types.
        
        Args:
            layer_types: List of layer types to include (e.g., ['mlp', 'self_attn'])
                        If None, returns all layers
        
        Returns:
            List of layer names
        """
        layer_names = []
        
        for name, module in self.model.named_modules():
            if layer_types is None:
                if len(list(module.children())) == 0:  # Leaf modules only
                    layer_names.append(name)
            else:
                for layer_type in layer_types:
                    if layer_type in name and len(list(module.children())) == 0:
                        layer_names.append(name)
                        break
        
        return layer_names
    
    def find_top_activating_neurons(self, 
                                   layer_name: str, 
                                   text: str, 
                                   top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Find neurons that activate most strongly for given text.
        
        Args:
            layer_name: Name of layer to analyze
            text: Input text
            top_k: Number of top neurons to return
            
        Returns:
            List of (neuron_index, activation_value, token) tuples
        """
        # Set up hooks for specific layer
        self.register_hooks([layer_name])
        
        # Analyze text
        results = self.analyze_text(text)
        activations = results[layer_name]['activations'][0]  # Shape: [seq_len, hidden_size]
        tokens = results[layer_name]['tokens']
        
        # Find top activating neurons across all positions
        top_neurons = []
        
        for pos, token in enumerate(tokens):
            pos_activations = activations[pos]  # Shape: [hidden_size]
            
            # Get top-k neurons for this position
            values, indices = torch.topk(pos_activations, top_k)
            
            for i, (neuron_idx, value) in enumerate(zip(indices, values)):
                top_neurons.append((neuron_idx.item(), value.item(), token, pos))
        
        # Sort by activation value and return top-k overall
        top_neurons.sort(key=lambda x: x[1], reverse=True)
        return top_neurons[:top_k]
    
    def compare_activations(self, 
                           texts: List[str], 
                           layer_name: str,
                           neuron_indices: List[int] = None) -> pd.DataFrame:
        """
        Compare how specific neurons respond to different texts.
        
        Args:
            texts: List of texts to compare
            layer_name: Layer to analyze
            neuron_indices: Specific neurons to track (if None, finds top activating)
            
        Returns:
            DataFrame with comparison results
        """
        self.register_hooks([layer_name])
        
        results = []
        
        for text in texts:
            analysis = self.analyze_text(text)
            activations = analysis[layer_name]['activations'][0]  # [seq_len, hidden_size]
            tokens = analysis[layer_name]['tokens']
            
            # If no specific neurons specified, find top activating ones
            if neuron_indices is None:
                # Average across all positions
                avg_activations = activations.mean(dim=0)
                _, top_indices = torch.topk(avg_activations, 20)
                neuron_indices = top_indices.tolist()
            
            # Extract activations for specified neurons
            for neuron_idx in neuron_indices:
                neuron_activations = activations[:, neuron_idx]
                max_activation = neuron_activations.max().item()
                mean_activation = neuron_activations.mean().item()
                
                # Find token with max activation
                max_pos = neuron_activations.argmax().item()
                max_token = tokens[max_pos] if max_pos < len(tokens) else ""
                
                results.append({
                    'text': text[:50] + "..." if len(text) > 50 else text,
                    'neuron_idx': neuron_idx,
                    'max_activation': max_activation,
                    'mean_activation': mean_activation,
                    'max_token': max_token,
                    'max_position': max_pos
                })
        
        return pd.DataFrame(results)
    
    def visualize_attention_patterns(self, 
                                   text: str, 
                                   layer_idx: int = 0, 
                                   head_idx: int = 0) -> None:
        """
        Visualize attention patterns for a specific layer and head.
        
        Args:
            text: Input text
            layer_idx: Layer index
            head_idx: Attention head index
        """
        # Hook the attention layer
        attn_layer_name = f"model.layers.{layer_idx}.self_attn"
        self.register_hooks([attn_layer_name])
        
        # We need to modify the hook to capture attention weights
        def attention_hook(module, input, output):
            # For Qwen2.5, output is (hidden_states, attention_weights)
            if len(output) > 1 and output[1] is not None:
                self.activations[attn_layer_name + "_attn"] = output[1].detach().cpu()
        
        # Register special attention hook
        attn_module = self._get_module_by_name(attn_layer_name)
        hook = attn_module.register_forward_hook(attention_hook)
        
        # Analyze text
        results = self.analyze_text(text)
        tokens = results[attn_layer_name]['tokens']
        
        # Get attention weights if available
        attn_key = attn_layer_name + "_attn"
        if attn_key in self.activations:
            attention_weights = self.activations[attn_key][0, head_idx]  # [seq_len, seq_len]
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attention_weights.numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            plt.title(f'Attention Patterns - Layer {layer_idx}, Head {head_idx}')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
        else:
            print("Attention weights not captured. Model might not output attention weights.")
        
        # Clean up
        hook.remove()
    
    def activation_patching_experiment(self, 
                                     clean_text: str,
                                     corrupted_text: str,
                                     layer_name: str,
                                     patch_positions: List[int] = None) -> Dict[str, float]:
        """
        Perform activation patching to test causal relationships.
        
        Args:
            clean_text: Text that produces correct behavior
            corrupted_text: Text that produces incorrect behavior  
            layer_name: Layer to patch
            patch_positions: Specific positions to patch (if None, patches all)
            
        Returns:
            Dictionary with results including performance recovery
        """
        self.register_hooks([layer_name])
        
        # Get activations for both texts
        clean_results = self.analyze_text(clean_text)
        corrupted_results = self.analyze_text(corrupted_text)
        
        clean_activations = clean_results[layer_name]['activations']
        corrupted_activations = corrupted_results[layer_name]['activations']
        
        # Get original predictions
        clean_logits = clean_results[layer_name]['logits']
        corrupted_logits = corrupted_results[layer_name]['logits']
        
        # Patch activations (replace corrupted with clean at specific positions)
        if patch_positions is None:
            patch_positions = list(range(corrupted_activations.shape[1]))
        
        patched_activations = corrupted_activations.clone()
        for pos in patch_positions:
            if pos < clean_activations.shape[1]:
                patched_activations[0, pos] = clean_activations[0, pos]
        
        # This is a simplified version - full implementation would require
        # running forward pass with patched activations
        
        return {
            "clean_performance": self._compute_next_token_accuracy(clean_logits, clean_text),
            "corrupted_performance": self._compute_next_token_accuracy(corrupted_logits, corrupted_text),
            "patch_positions": patch_positions,
            "note": "Full patching requires more complex intervention - this shows the framework"
        }
    
    def _compute_next_token_accuracy(self, logits: torch.Tensor, text: str) -> float:
        """Simplified performance metric - could be enhanced based on specific task"""
        # Get top prediction
        predicted_token_id = logits[0, -1].argmax().item()
        predicted_token = self.tokenizer.decode([predicted_token_id])
        
        # This is simplified - in practice you'd have specific tasks/metrics
        return float(len(predicted_token.strip()) > 0)  # Basic sanity check
    
    def neuron_feature_visualization(self, 
                                   layer_name: str, 
                                   neuron_idx: int,
                                   test_texts: List[str]) -> Dict[str, float]:
        """
        Understand what feature a specific neuron detects.
        
        Args:
            layer_name: Layer containing the neuron
            neuron_idx: Index of neuron to analyze
            test_texts: Various texts to test neuron response
            
        Returns:
            Dictionary mapping texts to neuron activation strength
        """
        self.register_hooks([layer_name])
        
        neuron_responses = {}
        
        for text in test_texts:
            results = self.analyze_text(text)
            activations = results[layer_name]['activations'][0]  # [seq_len, hidden_size]
            
            # Get max activation for this neuron across all positions
            neuron_activation = activations[:, neuron_idx].max().item()
            neuron_responses[text] = neuron_activation
        
        return neuron_responses

# Example usage and analysis functions
def demo_basic_analysis():
    """Demonstrate basic mechanistic interpretability analysis"""
    
    # Initialize interpreter
    interp = MechanisticInterpreter()
    
    # Example texts for analysis
    texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.", 
        "Mathematics is the language of science.",
        "I love eating pizza with cheese.",
        "The weather today is very sunny."
    ]
    
    print("\n=== Basic Activation Analysis ===")
    
    # Analyze MLP layers (these often contain interpretable features)
    mlp_layers = [name for name in interp.get_layer_names() if 'mlp' in name][:3]  # First 3 MLP layers
    interp.register_hooks(mlp_layers)
    
    for text in texts[:2]:  # Analyze first two texts
        print(f"\nAnalyzing: '{text}'")
        
        for layer in mlp_layers:
            top_neurons = interp.find_top_activating_neurons(layer, text, top_k=5)
            print(f"\nLayer {layer} - Top activating neurons:")
            for neuron_idx, activation, token, pos in top_neurons:
                print(f"  Neuron {neuron_idx}: {activation:.3f} (token: '{token}' at pos {pos})")
    
    print("\n=== Comparative Analysis ===")
    
    # Compare how neurons respond to different types of content
    if mlp_layers:
        comparison_df = interp.compare_activations(texts, mlp_layers[0])
        print("\nTop neurons across different texts:")
        print(comparison_df.groupby('neuron_idx')['max_activation'].mean().sort_values(ascending=False).head(10))
    
    print("\n=== Feature Detection Analysis ===")
    
    # Test what specific neurons might be detecting
    test_texts = [
        "cat", "dog", "animal", "the", "a", "an",  # Articles vs nouns
        "running", "jumping", "sitting",  # Action words
        "red", "blue", "green",  # Colors
        "123", "456", "789"  # Numbers
    ]
    
    if mlp_layers:
        # Pick a neuron that was highly active
        layer_name = mlp_layers[0]
        neuron_responses = interp.neuron_feature_visualization(layer_name, 100, test_texts)
        
        print(f"\nNeuron 100 in {layer_name} responses:")
        sorted_responses = sorted(neuron_responses.items(), key=lambda x: x[1], reverse=True)
        for text, activation in sorted_responses:
            print(f"  '{text}': {activation:.3f}")
    
    # Clean up
    interp.clear_hooks()
    
    return interp

if __name__ == "__main__":
    # Run the demo
    interpreter = demo_basic_analysis()
    print("\n=== Tool ready for your own experiments! ===")
    print("\nExample usage:")
    print("interpreter.register_hooks(['model.layers.0.mlp'])")
    print("results = interpreter.analyze_text('Your text here')")
    print("top_neurons = interpreter.find_top_activating_neurons('model.layers.0.mlp', 'Your text')")
    
    # Save this entire file as: mechanistic_interpreter.py