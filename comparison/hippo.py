import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--hippo_model_dir", type=str, required=True, help="Path to the hippo model directory")
    return parser.parse_args()

def generate_text(model, tokenizer, prompt, max_length=3000):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main(input_path, save_path, hippo_model_dir):
    # Load data from CSV
    df = pd.read_csv(input_path)

    # Specify the base model name
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"

    # Load the tokenizer and the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load the Hippo model
    hippo_model = PeftModel.from_pretrained(base_model, hippo_model_dir)

    # Iterate through each prompt in the dataframe
    for index, row in df.iterrows():
        prompt = row['prompt']

        # Generate the response using the model with explicit truncation
        response = generate_text(hippo_model, tokenizer, prompt)
        # Extract the answer from the response
        answer = response.split('Response:')[1].strip() if 'Response:' in response else response.strip()
        answer = answer.strip('"').strip("'")

        # Store the answer back in the dataframe
        df.at[index, 'hippo_answer'] = answer

    # Save the dataframe with answers to a new CSV file
    df.to_csv(save_path, index=False)
    print("Output saved to:", save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.input_path, args.save_path, args.hippo_model_dir)
