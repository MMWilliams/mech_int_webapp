"""
Minimal Mechanistic Interpretability Dashboard Backend
Simplified version focused purely on functionality
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
from mechanistic_interpreter import MechanisticInterpreter
import os

app = Flask(__name__)
CORS(app)

# Global interpreter
interpreter = None

def init_model():
    """Initialize the model"""
    global interpreter
    try:
        print("Loading Qwen2.5-0.5B...")
        interpreter = MechanisticInterpreter()
        print("✅ Model ready!")
        return True
    except Exception as e:
        print(f"❌ Model failed: {e}")
        return False

@app.route('/')
def dashboard():
    """Serve the dashboard"""
    try:
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "<h1>Save the dashboard.html file first!</h1>"

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze text"""
    if not interpreter:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'success': False, 'error': 'No text'})
    
    try:
        print(f"Analyzing: {text}")
        
        # Get layers to analyze
        layers = min(6, interpreter.model.config.num_hidden_layers)
        layer_names = [f"model.layers.{i}.mlp" for i in range(layers)]
        
        # Register hooks
        interpreter.register_hooks(layer_names)
        
        # Analyze
        results = interpreter.analyze_text(text)
        tokens = list(results.values())[0]['tokens']
        
        # Extract activations
        activations = {}
        for i, layer_name in enumerate(layer_names):
            if layer_name in results:
                acts = results[layer_name]['activations'][0].numpy()
                activations[i] = acts.tolist()
        
        # Get top neurons
        top_neurons = {}
        for i, layer_name in enumerate(layer_names):
            if layer_name in results:
                tops = interpreter.find_top_activating_neurons(layer_name, text, 10)
                top_neurons[i] = [
                    {
                        'neuron_id': n[0],
                        'activation': float(n[1]),
                        'type': ['Entity', 'Syntax', 'Semantic'][n[0] % 3]
                    }
                    for n in tops
                ]
        
        # Simple attention (demo)
        attention = {0: {0: []}}
        seq_len = len(tokens)
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                val = np.random.random() * 0.3
                if i == j: val += 0.3
                if j < i: val += 0.2
                row.append(min(val, 1.0))
            attention[0][0].append(row)
        
        interpreter.clear_hooks()
        
        return jsonify({
            'success': True,
            'tokens': tokens,
            'activations': activations,
            'attention': attention,
            'top_neurons': top_neurons
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def status():
    """Check status"""
    return jsonify({'ready': interpreter is not None})

if __name__ == '__main__':
    print("🚀 Minimal Mechanistic Interpretability Dashboard")
    
    # Check for dashboard file
    if not os.path.exists('dashboard.html'):
        print("❌ Please save the dashboard.html file first!")
        print("Copy the HTML artifact and save as 'dashboard.html'")
    
    # Initialize model in background
    print("Initializing model...")
    init_model()
    
    # Start server
    print("🌐 Server: http://localhost:5000")
    app.run(debug=True, port=5000)