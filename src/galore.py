# galore.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import caffe
import numpy as np

def check_cuda():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is not available.")

def load_model_and_tokenizer(model_name="gpt2"):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model loaded on CUDA")
    else:
        print("Model loaded on CPU")

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=max_length)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def setup_caffe_model(prototxt, caffemodel):
    # Set Caffe to GPU mode
    if torch.cuda.is_available():
        caffe.set_mode_gpu()
        print("Caffe set to GPU mode")
    else:
        caffe.set_mode_cpu()
        print("Caffe set to CPU mode")

    # Load the model
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net

def classify_with_caffe(net, input_data):
    # Assuming input_data is a numpy array of appropriate shape
    net.blobs['data'].data[...] = input_data
    output = net.forward()
    return output

if __name__ == "__main__":
    check_cuda()
    model_name = "gpt2"  # You can replace this with any large language model, e.g., "gpt-3", "t5-large", etc.
    model, tokenizer = load_model_and_tokenizer(model_name)

    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated Text: {generated_text}")

    # Setup and use Caffe model
    prototxt = 'path/to/deploy.prototxt'
    caffemodel = 'path/to/weights.caffemodel'
    caffe_net = setup_caffe_model(prototxt, caffemodel)

    # Example input data for Caffe (adjust shape as needed)
    input_data = np.random.rand(1, 3, 224, 224)  # Example shape for a typical image input
    caffe_output = classify_with_caffe(caffe_net, input_data)
    print(f"Caffe Model Output: {caffe_output}")
