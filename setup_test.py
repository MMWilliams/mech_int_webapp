#!/usr/bin/env python3
"""
Quick start script to verify your mechanistic interpretability setup
Run this after setting up your virtual environment to ensure everything works.
"""

def test_installation():
    """Test that all required packages are installed and working"""
    
    print("🔧 Testing installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found! Run: pip install torch")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} installed")
    except ImportError:
        print("❌ Transformers not found! Run: pip install transformers")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} installed")
    except ImportError:
        print("❌ NumPy not found! Run: pip install numpy")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} installed")
    except ImportError:
        print("❌ Matplotlib not found! Run: pip install matplotlib")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} installed")
    except ImportError:
        print("❌ Pandas not found! Run: pip install pandas")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__} installed")
    except ImportError:
        print("❌ Seaborn not found! Run: pip install seaborn")
        return False
    
    return True

def test_model_loading():
    """Test loading the Qwen model"""
    
    print("\n🤖 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        print("   Downloading model (this may take a few minutes on first run)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float32
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model has {model.config.num_hidden_layers} layers")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Attention heads: {model.config.num_attention_heads}")
        
        # Test basic inference
        test_text = "The cat sat on"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_id = outputs.logits[0, -1].argmax().item()
            predicted_token = tokenizer.decode([predicted_id])
        
        print(f"   Test inference: '{test_text}' -> '{predicted_token.strip()}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   Check your internet connection and try again")
        return False

def run_mini_demo():
    """Run a mini demonstration of the tool"""
    
    print("\n🎯 Running mini demo...")
    
    try:
        # Import our tool (assuming the files are in the same directory)
        try:
            from mechanistic_interpreter import MechanisticInterpreter
        except ImportError:
            print("❌ mechanistic_interpreter.py not found in current directory")
            print("   Make sure you've saved the main tool file as 'mechanistic_interpreter.py'")
            return False
        
        # Initialize interpreter
        print("   Initializing interpreter...")
        interp = MechanisticInterpreter()
        
        # Quick analysis
        print("   Running quick analysis...")
        
        # Get a few layer names
        mlp_layers = [name for name in interp.get_layer_names() if 'mlp' in name][:2]
        
        if mlp_layers:
            layer_name = mlp_layers[0]
            test_text = "The happy cat played with the red ball"
            
            print(f"   Analyzing: '{test_text}'")
            print(f"   Using layer: {layer_name}")
            
            # Find top neurons
            top_neurons = interp.find_top_activating_neurons(layer_name, test_text, top_k=3)
            
            print("   Top activating neurons:")
            for neuron_idx, activation, token, pos in top_neurons:
                print(f"     Neuron {neuron_idx}: {activation:.3f} (token: '{token}')")
        
        print("✅ Mini demo completed successfully!")
        interp.clear_hooks()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main setup verification function"""
    
    print("🚀 Mechanistic Interpretability Tool Setup Verification")
    print("=" * 60)
    
    # Test 1: Package installation
    if not test_installation():
        print("\n❌ Setup incomplete. Please install missing packages.")
        return
    
    # Test 2: Model loading
    if not test_model_loading():
        print("\n❌ Model loading failed. Check internet connection and try again.")
        return
    
    # Test 3: Tool demo
    if not run_mini_demo():
        print("\n❌ Tool demo failed. Make sure mechanistic_interpreter.py is in the current directory.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Your setup is complete and ready to use!")
    print("\n📚 Next steps:")
    print("   1. Run: python mechanistic_interpreter.py (for basic demo)")
    print("   2. Run: python advanced_examples.py (for circuit discovery)")
    print("   3. Start exploring with your own text and experiments!")
    print("\n💡 Remember to activate your virtual environment each time:")
    print("   source mech_interp_env/bin/activate  # Linux/Mac")
    print("   mech_interp_env\\Scripts\\activate     # Windows")

if __name__ == "__main__":
    main()
    
    # Save this entire file as: setup_test.py