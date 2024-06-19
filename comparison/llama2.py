import argparse
from transformers import pipeline
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output CSV file")
    return parser.parse_args()

def main(input_path, save_path):
    # Load data from CSV
    df = pd.read_csv(input_path)

    # Specify the model name
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # Load the Llama2 model pipeline with explicit truncation
    llama2_pl = pipeline("text-generation", model=model_name, tokenizer=model_name)

    # Iterate through each prompt in the dataframe
    for index, row in df.iterrows():
        prompt = row['prompt']

        # Generate the response using the model with explicit truncation
        response = llama2_pl(prompt, max_length=3000, truncation=True)
        # Extract the answer from the response
        answer = response[0]['generated_text'].split('Response:')[1].strip()
        answer = answer.strip('"').strip("'")

        # Store the answer back in the dataframe
        df.at[index, 'llama2_answer'] = answer

    # Save the dataframe with answers to a new CSV file
    df.to_csv(save_path, index=False)
    print("Output saved to:", save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.input_path, args.save_path)