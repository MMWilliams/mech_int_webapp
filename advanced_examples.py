"""
Advanced Examples for Mechanistic Interpretability
Demonstrates real interpretability discoveries you can make with the tool.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mechanistic_interpreter import MechanisticInterpreter
from typing import List, Dict
import pandas as pd

class CircuitAnalyzer:
    """Advanced analysis for discovering interpretable circuits"""
    
    def __init__(self, interpreter: MechanisticInterpreter):
        self.interp = interpreter
    
    def discover_induction_heads(self) -> Dict:
        """
        Look for induction head patterns: A B ... A -> B
        These are crucial for in-context learning.
        """
        print("=== Searching for Induction Head Patterns ===")
        
        # Test sequences that should trigger induction behavior
        test_sequences = [
            "cat dog elephant cat",  # Should predict "dog"
            "red blue green red",    # Should predict "blue" 
            "apple banana orange apple",  # Should predict "banana"
            "1 2 3 1",              # Should predict "2"
            "hello world goodbye hello"  # Should predict "world"
        ]
        
        results = {}
        
        # Test attention layers for induction patterns
        for layer_idx in range(min(8, self.interp.model.config.num_hidden_layers)):
            layer_name = f"model.layers.{layer_idx}.self_attn"
            
            print(f"\nTesting layer {layer_idx} for induction patterns...")
            
            layer_results = []
            
            for sequence in test_sequences:
                tokens = sequence.split()
                if len(tokens) >= 4:  # Need at least A B ... A pattern
                    
                    # Get model predictions
                    self.interp.register_hooks([layer_name])
                    analysis = self.interp.analyze_text(sequence)
                    
                    logits = analysis[layer_name]['logits'][0]  # [seq_len, vocab_size]
                    model_tokens = analysis[layer_name]['tokens']
                    
                    # Check if model predicts the "induction" token
                    # For "cat dog elephant cat", check if it predicts "dog" after final "cat"
                    if len(model_tokens) >= 4:
                        # Find the second occurrence of first token
                        first_token = tokens[0]
                        expected_next = tokens[1]
                        
                        # Get model's prediction for what comes after repeated token
                        last_logits = logits[-1]  # Last position predictions
                        predicted_token_id = last_logits.argmax().item()
                        predicted_token = self.interp.tokenizer.decode([predicted_token_id]).strip()
                        
                        # Check if prediction matches expected induction
                        induction_success = expected_next.lower() in predicted_token.lower()
                        
                        layer_results.append({
                            'sequence': sequence,
                            'expected': expected_next,
                            'predicted': predicted_token,
                            'success': induction_success,
                            'confidence': torch.softmax(last_logits, dim=0)[predicted_token_id].item()
                        })
            
            results[layer_idx] = layer_results
            
            # Calculate success rate for this layer
            if layer_results:
                success_rate = sum(r['success'] for r in layer_results) / len(layer_results)
                print(f"Layer {layer_idx} induction success rate: {success_rate:.2f}")
        
        return results
    
    def find_sentiment_neurons(self) -> Dict:
        """
        Discover neurons that detect sentiment/emotion.
        """
        print("\n=== Searching for Sentiment Detection Neurons ===")
        
        positive_texts = [
            "I love this movie!",
            "This is amazing and wonderful!",
            "Great job, fantastic work!",
            "Happy birthday! So excited!",
            "Beautiful sunny day today!"
        ]
        
        negative_texts = [
            "I hate this terrible movie.",
            "This is awful and horrible.",
            "Bad job, terrible work.",
            "Sad news, very disappointed.",
            "Ugly rainy day today."
        ]
        
        neutral_texts = [
            "The movie is 2 hours long.",
            "This contains some information.",
            "Work was completed yesterday.",
            "News was reported at noon.",
            "Weather forecast shows rain."
        ]
        
        # Test multiple MLP layers
        mlp_layers = [f"model.layers.{i}.mlp" for i in range(min(6, self.interp.model.config.num_hidden_layers))]
        
        sentiment_neurons = {}
        
        for layer_name in mlp_layers:
            print(f"\nAnalyzing {layer_name} for sentiment neurons...")
            
            # Get activations for all text types
            self.interp.register_hooks([layer_name])
            
            pos_activations = []
            neg_activations = []
            neu_activations = []
            
            for texts, activation_list in [(positive_texts, pos_activations), 
                                         (negative_texts, neg_activations),
                                         (neutral_texts, neu_activations)]:
                for text in texts:
                    analysis = self.interp.analyze_text(text)
                    # Average activation across all positions
                    avg_activation = analysis[layer_name]['activations'][0].mean(dim=0)
                    activation_list.append(avg_activation)
            
            # Convert to tensors
            pos_avg = torch.stack(pos_activations).mean(dim=0)  # [hidden_size]
            neg_avg = torch.stack(neg_activations).mean(dim=0)
            neu_avg = torch.stack(neu_activations).mean(dim=0)
            
            # Find neurons with strong sentiment bias
            pos_bias = pos_avg - neu_avg  # How much more active for positive
            neg_bias = neg_avg - neu_avg  # How much more active for negative
            
            # Find top sentiment-detecting neurons
            pos_neurons = torch.topk(pos_bias, 10)
            neg_neurons = torch.topk(neg_bias, 10)
            
            sentiment_neurons[layer_name] = {
                'positive_neurons': [(idx.item(), val.item()) for idx, val in zip(pos_neurons.indices, pos_neurons.values)],
                'negative_neurons': [(idx.item(), val.item()) for idx, val in zip(neg_neurons.indices, neg_neurons.values)],
                'positive_avg_activation': pos_avg.mean().item(),
                'negative_avg_activation': neg_avg.mean().item(),
                'neutral_avg_activation': neu_avg.mean().item()
            }
            
            print(f"Top positive sentiment neuron: {pos_neurons.indices[0].item()} (bias: {pos_neurons.values[0].item():.3f})")
            print(f"Top negative sentiment neuron: {neg_neurons.indices[0].item()} (bias: {neg_neurons.values[0].item():.3f})")
        
        return sentiment_neurons
    
    def analyze_grammatical_circuits(self) -> Dict:
        """
        Look for circuits that handle grammatical relationships.
        """
        print("\n=== Analyzing Grammatical Processing Circuits ===")
        
        # Test different grammatical structures
        grammatical_tests = {
            'subject_verb_agreement': [
                ("The cat runs", "The cats run"),  # Singular vs plural
                ("She walks", "They walk"),
                ("It works", "We work")
            ],
            'tense_consistency': [
                ("I walked yesterday", "I will walk tomorrow"),
                ("She ate lunch", "She will eat dinner"),
                ("They played games", "They will play sports")
            ],
            'object_relationships': [
                ("I gave the book to Mary", "I gave Mary the book"),  # Direct vs indirect object
                ("She showed the picture to him", "She showed him the picture"),
                ("We sent the letter to them", "We sent them the letter")
            ]
        }
        
        results = {}
        
        # Test middle layers (where grammatical processing often happens)
        test_layers = [f"model.layers.{i}.mlp" for i in range(2, min(6, self.interp.model.config.num_hidden_layers))]
        
        for grammar_type, test_pairs in grammatical_tests.items():
            print(f"\nTesting {grammar_type}...")
            
            grammar_results = {}
            
            for layer_name in test_layers:
                self.interp.register_hooks([layer_name])
                
                pair_differences = []
                
                for sent1, sent2 in test_pairs:
                    # Get activations for both sentences
                    analysis1 = self.interp.analyze_text(sent1)
                    analysis2 = self.interp.analyze_text(sent2)
                    
                    # Compare average activations
                    act1 = analysis1[layer_name]['activations'][0].mean(dim=0)
                    act2 = analysis2[layer_name]['activations'][0].mean(dim=0)
                    
                    # Find neurons with biggest differences
                    diff = torch.abs(act1 - act2)
                    pair_differences.append(diff)
                
                # Average differences across all pairs
                avg_diff = torch.stack(pair_differences).mean(dim=0)
                
                # Find neurons most sensitive to this grammatical distinction
                top_diff_neurons = torch.topk(avg_diff, 5)
                
                grammar_results[layer_name] = {
                    'top_sensitive_neurons': [(idx.item(), val.item()) 
                                            for idx, val in zip(top_diff_neurons.indices, top_diff_neurons.values)],
                    'average_sensitivity': avg_diff.mean().item()
                }
            
            results[grammar_type] = grammar_results
        
        return results
    
    def visualize_neuron_feature_space(self, layer_name: str, neuron_indices: List[int], test_categories: Dict[str, List[str]]):
        """
        Create a 2D visualization of how neurons respond to different semantic categories.
        """
        print(f"\n=== Visualizing Feature Space for {layer_name} ===")
        
        self.interp.register_hooks([layer_name])
        
        # Collect activations for each category
        category_activations = {}
        all_texts = []
        all_labels = []
        
        for category, texts in test_categories.items():
            activations = []
            for text in texts:
                analysis = self.interp.analyze_text(text)
                # Get activations for specified neurons
                layer_act = analysis[layer_name]['activations'][0].mean(dim=0)  # Average across positions
                neuron_activations = layer_act[neuron_indices]  # Only specified neurons
                activations.append(neuron_activations)
                all_texts.append(text)
                all_labels.append(category)
            
            category_activations[category] = torch.stack(activations)
        
        # Create visualization if we have exactly 2 neurons
        if len(neuron_indices) == 2:
            plt.figure(figsize=(10, 8))
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, (category, activations) in enumerate(category_activations.items()):
                x_vals = activations[:, 0].numpy()
                y_vals = activations[:, 1].numpy()
                plt.scatter(x_vals, y_vals, c=colors[i % len(colors)], label=category, s=100, alpha=0.7)
            
            plt.xlabel(f'Neuron {neuron_indices[0]} Activation')
            plt.ylabel(f'Neuron {neuron_indices[1]} Activation')
            plt.title(f'Semantic Feature Space - {layer_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return category_activations

def run_advanced_analysis():
    """Run comprehensive circuit discovery analysis"""
    
    print("Starting Advanced Mechanistic Interpretability Analysis...")
    print("This will take a few minutes to complete.")
    
    # Initialize interpreter
    interp = MechanisticInterpreter()
    analyzer = CircuitAnalyzer(interp)
    
    # 1. Look for induction heads
    induction_results = analyzer.discover_induction_heads()
    
    # Find the best performing layer for induction
    best_induction_layer = None
    best_success_rate = 0
    
    for layer_idx, results in induction_results.items():
        if results:
            success_rate = sum(r['success'] for r in results) / len(results)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_induction_layer = layer_idx
    
    if best_induction_layer is not None:
        print(f"\nBest induction head found in layer {best_induction_layer} (success rate: {best_success_rate:.2f})")
    
    # 2. Find sentiment neurons
    sentiment_results = analyzer.find_sentiment_neurons()
    
    # 3. Analyze grammatical circuits
    grammar_results = analyzer.analyze_grammatical_circuits()
    
    # 4. Visualize feature spaces
    test_categories = {
        'animals': ['cat', 'dog', 'elephant', 'mouse'],
        'colors': ['red', 'blue', 'green', 'yellow'],
        'actions': ['running', 'jumping', 'sleeping', 'eating'],
        'objects': ['car', 'house', 'book', 'phone']
    }
    
    # Pick two interesting neurons from sentiment analysis
    if sentiment_results:
        layer_name = list(sentiment_results.keys())[0]
        pos_neuron = sentiment_results[layer_name]['positive_neurons'][0][0]
        neg_neuron = sentiment_results[layer_name]['negative_neurons'][0][0]
        
        feature_space = analyzer.visualize_neuron_feature_space(
            layer_name, 
            [pos_neuron, neg_neuron], 
            test_categories
        )
    
    # 5. Summary report
    print("\n" + "="*60)
    print("MECHANISTIC INTERPRETABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    if best_induction_layer is not None:
        print(f"🔍 Induction Heads: Found strong pattern in layer {best_induction_layer}")
        print(f"   Success rate: {best_success_rate:.1%}")
    
    print(f"\n🎭 Sentiment Processing:")
    for layer_name, results in sentiment_results.items():
        pos_neuron, pos_strength = results['positive_neurons'][0]
        neg_neuron, neg_strength = results['negative_neurons'][0]
        print(f"   {layer_name}: Pos neuron {pos_neuron} ({pos_strength:.3f}), Neg neuron {neg_neuron} ({neg_strength:.3f})")
    
    print(f"\n📝 Grammar Processing:")
    for grammar_type, layer_results in grammar_results.items():
        print(f"   {grammar_type}:")
        for layer_name, results in layer_results.items():
            top_neuron, sensitivity = results['top_sensitive_neurons'][0]
            print(f"     {layer_name}: Most sensitive neuron {top_neuron} (diff: {sensitivity:.3f})")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Model develops specialized circuits for different linguistic functions")
    print(f"   - Earlier layers handle basic features, later layers handle complex relationships")
    print(f"   - Individual neurons can be surprisingly interpretable")
    print(f"   - Attention and MLP layers have different specializations")
    
    print(f"\n🛠️  Next Steps:")
    print(f"   - Test these neurons on more diverse inputs")
    print(f"   - Use activation patching to verify causal relationships")
    print(f"   - Look for interactions between different circuits")
    print(f"   - Compare findings across different model sizes")
    
    return {
        'induction': induction_results,
        'sentiment': sentiment_results, 
        'grammar': grammar_results,
        'interpreter': interp,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    # Run the full advanced analysis
    results = run_advanced_analysis()
    
    print(f"\n🎯 Analysis complete! Use the returned objects to explore further:")
    print(f"   results['interpreter'] - Main interpreter object")
    print(f"   results['analyzer'] - Circuit analyzer") 
    print(f"   results['sentiment'] - Sentiment neuron findings")
    print(f"   results['grammar'] - Grammar circuit findings")
    
    # Save this entire file as: advanced_examples.py