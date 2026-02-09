"""
Dataset Inspector - Check the actual structure of your HuggingFace dataset
"""

from datasets import load_dataset
import json

print("="*80)
print("🔍 DATASET INSPECTOR")
print("="*80)

dataset_name = "yusufbukarmaina/Beakers1"

print(f"\n📦 Loading dataset: {dataset_name}")
print("This may take a moment...\n")

try:
    # Load dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    # Get first 5 examples
    examples = []
    for i, example in enumerate(ds):
        examples.append(example)
        if i >= 4:  # Get 5 examples (0-4)
            break
    
    print("="*80)
    print(f"✅ Successfully loaded dataset!")
    print("="*80)
    
    # Show structure of first example
    print(f"\n📋 EXAMPLE 1 - Full Structure:")
    print("-"*80)
    
    first_example = examples[0]
    print(f"Available fields: {list(first_example.keys())}\n")
    
    for key, value in first_example.items():
        print(f"Field: '{key}'")
        print(f"  Type: {type(value).__name__}")
        
        if key == 'image':
            print(f"  Value: <PIL Image object>")
            if hasattr(value, 'size'):
                print(f"  Size: {value.size}")
        elif isinstance(value, (str, int, float)):
            print(f"  Value: {repr(value)[:200]}")
        elif isinstance(value, dict):
            print(f"  Value: {json.dumps(value, indent=4)[:200]}")
        elif isinstance(value, list):
            print(f"  Value: List with {len(value)} items")
            if len(value) > 0:
                print(f"  First item: {repr(value[0])[:100]}")
        else:
            print(f"  Value: {str(value)[:200]}")
        
        print()
    
    # Show all 5 examples in compact form
    print("\n" + "="*80)
    print("📊 FIRST 5 EXAMPLES - Compact View")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print("-"*40)
        for key, value in example.items():
            if key == 'image':
                print(f"  {key}: <Image>")
            elif isinstance(value, str):
                # Truncate long strings
                display_value = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {repr(display_value)}")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}: {json.dumps(value)[:100]}...")
            elif isinstance(value, list):
                print(f"  {key}: [list with {len(value)} items]")
            else:
                print(f"  {key}: {str(value)[:100]}")
    
    # Analysis
    print("\n" + "="*80)
    print("🔬 ANALYSIS")
    print("="*80)
    
    # Check for required fields
    has_image = 'image' in first_example
    
    volume_fields = ['volume', 'answer', 'label', 'text', 'caption', 'conversations', 'messages']
    found_volume_fields = [f for f in volume_fields if f in first_example]
    
    print(f"\n✓ Has 'image' field: {has_image}")
    print(f"✓ Found volume-related fields: {found_volume_fields if found_volume_fields else 'None'}")
    
    print("\n📝 Recommendations:")
    
    if not has_image:
        print("  ❌ Missing 'image' field - dataset cannot be used for vision models")
    
    if not found_volume_fields:
        print("  ❌ No volume information found in any expected field")
        print("  💡 Available fields are:", list(first_example.keys()))
        print("  💡 Please identify which field contains the volume information")
    else:
        print(f"  ✅ Volume info might be in: {found_volume_fields}")
        print(f"\n  📋 Content of volume fields:")
        for field in found_volume_fields:
            value = first_example[field]
            if isinstance(value, str):
                print(f"     {field}: {repr(value[:200])}")
            else:
                print(f"     {field}: {value}")
    
    # Dataset format detection
    print("\n" + "="*80)
    print("🎯 DETECTED FORMAT")
    print("="*80)
    
    if 'conversations' in first_example or 'messages' in first_example:
        print("  Format: Chat/Conversation dataset")
        conv_field = 'conversations' if 'conversations' in first_example else 'messages'
        print(f"  Field: {conv_field}")
        print(f"  Content: {first_example[conv_field]}")
    elif 'text' in first_example:
        print("  Format: Image-Text pairs")
        print(f"  Text content: {first_example['text'][:200]}")
    elif 'label' in first_example or 'answer' in first_example:
        print("  Format: Image-Label pairs")
        field = 'label' if 'label' in first_example else 'answer'
        print(f"  Label/Answer: {first_example[field]}")
    else:
        print("  Format: Custom/Unknown")
        print("  Please manually inspect the fields above")
    
    print("\n" + "="*80)
    print("💡 NEXT STEPS")
    print("="*80)
    print("\nBased on the inspection above, you need to:")
    print("1. Identify which field contains the volume information")
    print("2. Update the training script to read from that field")
    print("3. Ensure the volume can be extracted (e.g., '250 mL' or just '250')")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print("\nPossible issues:")
    print("1. Dataset doesn't exist or is private")
    print("2. Not logged into HuggingFace (run: huggingface-cli login)")
    print("3. Network connection issues")
    
    import traceback
    print("\nFull error:")
    traceback.print_exc()

print("\n" + "="*80)
