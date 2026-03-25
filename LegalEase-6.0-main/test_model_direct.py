"""Simple test to check if the model can be loaded directly"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Test loading the model directly"""
    print("=== Testing Direct Model Loading ===")

    model_name = 'facebook/opt-350m'

    try:
        print(f"1. Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")

        print(f"2. Loading model for {model_name}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Model device: {model.device}")

        print("3. Testing inference...")
        test_text = "What is the capital of France?"
        inputs = tokenizer(test_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✓ Inference successful")
        print(f"  Response: {response}")

        return True

    except Exception as e:
        print(f"✗ Error during model loading/testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading and inference working!")
    else:
        print("\n❌ Model has issues.")
