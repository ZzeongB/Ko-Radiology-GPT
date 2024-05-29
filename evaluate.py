"""
This code is modification of asclepius: https://github.com/starmpcc/Asclepius
"""

import argparse
import random
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from secrets import OPENAI_API_KEY

# Constants
MODEL_NAME = "gpt-4"
MAX_TOKENS = 2048
TEMPERATURE = 0

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

prompt_acc = """You are an intelligent clinical language model. 

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

{answers}
Above, we provide you a radiology report and the question that the healthcare professional gave about the radiology report.
You are also provided with {num_samples} corresponding responses from {num_samples} different clinical models.
Your task is to compare each model's responses and evaluate the response's accuracy based on the following criteria.

Criteria : 
1. Unacceptable (1 point): The model's response contains numerous inaccuracies, including incorrect information, misinterpretations, or factual errors that significantly impact the reliability of the output.
2. Poor (2 points): The model's response contains incorrect information that differs from the provided report.
3. Satisfactory (3 points): The model's response is generally accurate but may have unclear or unnecessary information.
4. Excellent (4 points): The model's response is entirely accurate, providing information that is correct. Any unclear or unnecessary explanations have been removed from the answer. 

When evaluating each score based on above criteria, ensure that each judgement is not affected by other model's response.
First line must contain only {num_samples} values, which indicate the score for each model, respectively.
The {num_samples} scores are separated by a space.
Output scores without explanation.
"""

prompt_coc = """You are an intelligent language model. 

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

{answers}
Above, we provide you a radiology report and the question about the radiology report.
You are also provided with {num_samples} corresponding responses from {num_samples} different language models.
Your task is to compare each model's responses and evaluate the response's conciseness based on the following criteria.
Caution! You must valuate only the brevity of contents.

Criteria:
1. Unacceptable (1 point): The model's response is too long or totally inaccurate.
2. Poor (2 points): The model's answer is somewhat correct, but the explanation is too long. There are sentences that can be deleted, or sentences with the same content can be changed to be shorter.
3. Satisfactory (3 points): The model's answer is correct. But there is a shorter way to explain the same thing.
4. Excellent (4 points): The model's answer is short and precise. It does not give long-winded explanations or mention unnecessary details.

When evaluating each score based on above criteria, ensure that each judgement is not affected by other model's response.
First line must contain only {num_samples} values, which indicate the score for each model, respectively.
The {num_samples} scores are separated by a space.
Output scores without explanation.
"""

prompt_und = """You are an intelligent language model. 

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

{answers}
Above, we provide you a radiology report and the question about the radiology report.
You are also provided with {num_samples} corresponding responses from {num_samples} different language models.
Your task is to compare each model's responses and evaluate the response's understandability based on the following criteria.

Criteria:
1. Unacceptable (1 point): The model's answers mostly consist of medical terms that are difficult for non-medical experts to interpret.
2. Poor (2 points): The model's answers are difficult for non-medical experts to understand. Even though there are difficult medical terms, no explanation is provided.
3. Satisfactory (3 points): The model's answers can be understood to some extent by non-medical experts. Some difficult medical terms are explained.
4. Excellent (4 points): The model's answers can be easily understood even by non-medical experts. If there is difficult medical terminology in the answer, an explanation is also provided.

When evaluating each score based on above criteria, ensure that each judgement is not affected by other model's response.
First line must contain only {num_samples} values, which indicate the score for each model, respectively.
The {num_samples} scores are separated by a space.
Output scores without explanation.
"""

prompt_cos = """You are an intelligent language model. 

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

{answers}
Above, we provide you a radiology report and the question about the radiology report.
You are also provided with {num_samples} corresponding responses from {num_samples} different language models.
Your task is to compare each model's responses and evaluate the response's consistency based on the following criteria.

Criteria:
1. Unacceptable (1 point): The model's answers mostly consist of medical terms that are difficult for non-medical experts to interpret.
2. Poor (2 points): The model's answers are difficult for non-medical experts to understand. Even though there are difficult medical terms, no explanation is provided.
3. Satisfactory (3 points): The model's answers can be understood to some extent by non-medical experts. Some difficult medical terms are explained.
4. Excellent (4 points): The model's answers can be easily understood even by non-medical experts. If there is difficult medical terminology in the answer, an explanation is also provided.

When evaluating each score based on above criteria, ensure that each judgement is not affected by other model's response.
First line must contain only {num_samples} values, which indicate the score for each model, respectively.
The {num_samples} scores are separated by a space.
Output scores without explanation.
"""


def generate_prompt(type, report, question, samples):
    answers = ""
    for i, sample in enumerate(samples):
        sample_name = chr(65 + i)
        answers += f"[Agent {sample_name}'s Answer Begin]\n{sample}\n[Agent {sample_name}'s Answer End]\n\n"
    if type == "acc":  # accuracy
        return [
            {
                "role": "user",
                "content": prompt_acc.format(
                    report=report, question=question, answers=answers, num_samples=len(samples)
                ),
            }
        ]
    elif type == "coc":  # conciseness
        return [
            {
                "role": "user",
                "content": prompt_coc.format(
                    report=report, question=question, answers=answers, num_samples=len(samples)
                ),
            }
        ]
    elif type == "und":  # understandability
        return [
            {
                "role": "user",
                "content": prompt_und.format(
                    report=report, question=question, answers=answers, num_samples=len(samples)
                ),
            }
        ]
    elif type == "cos":  # consistency
        return [
            {
                "role": "user",
                "content": prompt_cos.format(
                    report=report, question=question, answers=answers, num_samples=len(samples)
                ),
            }
        ]


def ask_gpt(messages):
    for i in range(10):
        try:
            response = client.chat.completions.create(
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                model=MODEL_NAME,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
    return str(response)


def save_results(results, output_file):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, mode='a', index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--type", dest="type", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input_path)
    for i in data.columns:
        if "Unnamed" in i:
            data = data.rename(columns={"Unnamed: 0": "index"})
    answer_cols = [i for i in data.columns if "answer" in i]

    all_results = []

    for _, row in tqdm(data.iterrows()):
        order = list(range(len(answer_cols)))
        random.shuffle(order)

        report = row["report"]
        question = row["question"]
        samples = row[answer_cols].values[order]

        prompt = generate_prompt(args.type, report, question, samples)
        answer = ask_gpt(prompt)
        answer = answer.strip('"').strip("'")
        splitted_answer = answer.split()

        try:
            [splitted_answer[order.index(idx)] for idx in range(len(answer_cols))]
        except:
            for idx, col in enumerate(answer_cols):
                model_name = "_".join(col.split("_")[:-1])
                row[f"{model_name}_score"] = 0
        else:
            for idx, col in enumerate(answer_cols):
                model_name = "_".join(col.split("_")[:-1])
                row[f"{model_name}_score"] = splitted_answer[order.index(idx)]

        row["gpt4_response"] = answer
        all_results.append(row.to_dict())

    save_results(all_results, args.save_path)


if __name__ == "__main__":
    main()
