"""
Source: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
NOTE: Note/Inst gen usually bounded on TPM, not RPM
thus, simply ignore rpm cases
Also, the file size can be loaded into mem
"""

##TODO: Checklist
## 1. API KEY 확인
## 2. Output 주소 확인
## 3. Note Index 제대로 들어갔는지 확인
## 4. Max_min_token 확인
## 5. Task 별 비율 확인

import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import random
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata

# imports
import aiohttp  # for making API calls concurrently
import pandas as pd

global_prompt = """As a healthcare professional, you frequently employ a clinical language model to conduct thorough analyses of radiology reports. You ask the model some questions based on the radiology report, and get answers that helps your decision-making. 

Let's suppose you want to ask a question about the task of "{task}" related to the given radiology report. What specific question would you formulate?

[Radiology Report Begin]
{note}
[Radiology Report End]

The subsequent questions serve as examples for reference. You can freely refer to the examples, but try not to just replicate them. Your question should be answerable solely based on the information available in the radiology report. Your question must be an appropriate question for the task presented above.

[Example Questions Begin]
{samples}
[Example Questions End]

Please speak in Korean.
I encourage you to generate a question that is clear, novel, and analytically important. 
You must output solely a question.

Question:"""

tasks = [
    {
        "task": "질문 답변",
        "samples": [
            "CXR 보고서에 따르면 복부에 감염된 증거가 있습니까?",
            "흉부 엑스레이 결과 환자가 질병을 가지고 있는지 없는지 알 수 있습니까? 그렇다면 어떤 질병을 가지고 있었습니까?",
            "CXR 보고서에서 환자에게 삽관되는 기관내관과 구강위관은 어디에 있습니까?",
            "흉부 엑스레이의 내용으로 볼 때, 이 환자에게 어떤 치료가 필요합니까?",
            "이 환자의 심장과 폐의 상태는 어떻습니까? 이 환자는 기흉이 있습니까?",
        ],
        "pool": [],
    },
    {
        "task": "축약어 확장",
        "samples": [
            "방사선의학 보고서에 사용된 약어 'PA'의 전체 형태는 무엇입니까?",
            "'AP'의 의미를 이 보고서의 맥락에서 설명해 주십시오.",
            "'SOB'라는 약어의 확장 가능성은?",
            "방사선의학 보고서에서 'CT'는 무엇을 가리키나요?",
            "방사선의학 보고서에서 COPD라는 약어의 의미는 무엇입니까?",
        ],
        "pool": [],
    },
    {
        "task": "바꾸어 말하기",
        "samples": [
            "영상의학 보고서에는 '우심방에서 끝나는 카테터 팁'이 언급되어 있습니다. 환자가 더 쉽게 이해할 수 있도록 이것을 어떻게 다시 표현하시겠습니까?",
            "환자의 이해력 향상을 위해 보고서에 사용된 '좌측 기저 무기폐'라는 용어를 단순화할 수 있습니까?",
            "'앞서 설명한 오른쪽 기관 편차는 현재 연구에서 보이지 않습니다'라는 문구를 환자가 더 쉽게 이해할 수 있는 방법으로 어떻게 다시 표현하시겠습니까?",
            "보고서에 언급된 것처럼 '중앙 흉골 전선은 손상되지 않았습니다'에 대한 덜 임상적인 비유를 제공해 주시겠습니까?",
            "'위식도접합부 근위측 포트로 위의 비위관 팁, 2~4cm 정도 진전을 시사한다'는 문구를 환자가 쉽게 이해할 수 있는 언어로 바꿔주세요.",
        ],
        "pool": [],
    },
    {
        "task": "요약",
        "samples": [
            "이 보고서에 언급된 주요 방사선 소견에 대해 간략하게 요약해 주시겠습니까?",
            "보고된 바와 같이 환자의 가슴 상태 및 튜브 배치와 관련하여 주목할 만한 관찰 사항을 식별하고 요약합니다.",
            "방사선의학 보고서를 감안하여 폐 질환과 관련된 주요 소견과 우려되는 부분을 추출하여 요약해 주시겠습니까?",
            "방사선의학 보고서의 내용을 최대한 간결하게 정리해 주시기 바랍니다.",
            "이 환자가 흉부 엑스레이를 받은 이유와 검사 결과를 간단히 정리해 주시기 바랍니다.",
        ],
        "pool": [],
    },
]


# tasks = [
#     {
#         "task": "Question Answering",
#         "samples": [
#             "According to the CXR report, is there any evidence of infection in the abdomen?",
#             "Do chest x-ray results reveal whether a patient has a disease or not? If so, what disease did you have?",
#             "In the CXR report, where are the endotracheal tube and orogastric tube that are intubated in the patient?",
#             "Based on the contents of the chest X-ray, what treatment is needed for this patient?",
#             "What is the condition of this patient's heart and lungs? Does this patient have a pneumothorax?",
#         ],
#         "pool": [],
#     },
#     {
#         "task": "Abbreviation Expansion",
#         "samples": [
#             "What is the full form of the abbreviation 'PA' as used in the radiology report?",
#             "Please explain the meaning of 'AP' in the context of this report.",
#             "What is the possible expansion of the abbreviation 'SOB'?",
#             "In the radiology report, what does 'CT' refer to?",
#             "What is the meaning of the abbreviation COPD in the radiology report?"
#         ],
#         "pool": [],
#     },
#     {
#         "task": "Paraphrasing",
#         "samples": [
#             "The radiology report mentions 'catheter tip terminating in the right atrium.' How would you rephrase this to make it more understandable for the patient?",
#             "Can you simplify the term 'left basilar atelectasis' as used in the report for better patient comprehension?",
#             "How would you rephrase the phrase 'previously described right tracheal deviation is not seen on the current study' in a way that is more easily understood by the patient?",
#             "Could you provide a less clinical paraphrase for 'median sternal wires are intact' as mentioned in the report?",
#             "Translate the statement 'Nasogastric tube tip in the stomach with the proximal side port at the gastroesophageal junction, suggest advancement by 2-4cm' into language that a patient can easily understand.",
#         ],
#         "pool": [],
#     },
#     {
#         "task": "Summarization",
#         "samples": [
#             "Can you provide a concise summary of the key radiological findings mentioned in this report?",
#             "Identify and summarize any notable observations related to the patient's chest condition and tube placements as reported.",
#             "Given the radiology report, could you extract and summarize the main findings concerning lung conditions and any areas of concern?",
#             "Please summarize the contents of the radiology report as concisely as possible.",
#             "Please briefly summarize why this patient had a chest X-ray and the test results.",
#         ],
#         "pool": [],
#     },
# ]


@dataclass
class APICaller:
    requests_filepath: str
    save_filepath: str
    request_url: str
    api_key: str
    max_requests_per_minute: float
    max_tokens_per_minute: float
    token_encoding_name: str
    max_attempts: int
    logging_level: int

    def __post_init__(self):
        self.tasks = tasks
        self.sema = asyncio.Semaphore(1)
        self.rate_limit_sleep = False
        self.api_endpoint = "chat/completions"
        if "azure" in self.request_url:
            self.request_headers = {"api-key": self.api_key}
        else:
            self.request_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.task_id_generator = task_id_generator_function()

    async def run(self):
        df = pd.read_csv(self.requests_filepath)
        if "TEXT" in df.columns:
            df.rename(columns={"TEXT": "note"}, inplace=True)
        df["idx"] = df.apply(
            lambda x: random.choices([0, 1, 2, 3], weights=[0.5, 0.0, 0.25, 0.25])[0],
            axis=1,
        )
        data = df[["note", "idx"]].to_dict(orient="records")
        ##tasks = [self.await_and_call(i) for i in data[20000:30000]]
        tasks = [self.await_and_call(i) for i in data[60000:70000]]
        await asyncio.gather(*tasks)

    async def await_and_call(self, request):
        """
        request: {"note":note text, "idx": task_idx}
        """
        async with self.sema:
            if self.rate_limit_sleep:
                await asyncio.sleep(10)
                self.rate_limit_sleep = False
            await asyncio.sleep(
                2048 * 60 / self.max_tokens_per_minute
            )  # Max 4000, but actually about ~2000
        task_id = next(self.task_id_generator)
        samples = random.sample(self.tasks[request["idx"]]["samples"], 3)
        pool_num_samples = min(len(self.tasks[request["idx"]]["pool"]), 2)
        samples += random.sample(self.tasks[request["idx"]]["pool"], pool_num_samples)
        samples = "\n".join(samples)

        prompt = global_prompt.format(
            task=self.tasks[request["idx"]]["task"],
            samples=samples,
            note=request["note"],
        )

        request_json = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # Max of GPT-3.5-Turbo
            "max_tokens": 300,
            "temperature": 1,
        }

        res = await self.call_api(task_id, request_json)
        try:
            self.tasks[request["idx"]]["pool"].append(
                res["choices"][0]["message"]["content"]
            )
        except:
            # If Content Filtered
            pass

    async def call_api(
        self,
        task_id: int,
        request_json: dict,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{task_id}")
        for _ in range(self.max_attempts):
            error = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.request_url,
                        headers=self.request_headers,
                        json=request_json,
                    ) as response:
                        response = await response.json(content_type=None)
                if "error" in response:
                    logging.warning(
                        f"Request {task_id} failed with error {response['error']}"
                    )
                    error = response
                    if "Rate limit" in response["error"].get("message", ""):
                        self.rate_limit_sleep = True
                else:
                    break
            except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
                logging.warning(f"Request {task_id} failed with Exception {e}")
                error = e
        if error:
            logging.error(
                f"Request {request_json} failed after all attempts. Saving errors"
            )
            append_to_jsonl([request_json, [str(error)]], self.save_filepath)
            return None
        else:
            append_to_jsonl([request_json, response], self.save_filepath)
            logging.debug(f"Request {task_id} saved to {self.save_filepath}")
            return response


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script
async def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--save_path")
    parser.add_argument(
        "--request_url", default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument("--api_key")
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.input_path.replace(".csv", "_results.jsonl")

    caller = APICaller(
        requests_filepath=args.input_path,
        save_filepath=args.save_path,
        request_url=args.request_url,
        api_key=args.api_key,
        max_requests_per_minute=float(args.max_requests_per_minute),
        max_tokens_per_minute=float(args.max_tokens_per_minute),
        token_encoding_name=args.token_encoding_name,
        max_attempts=int(args.max_attempts),
        logging_level=int(args.logging_level),
    )
    await caller.run()


if __name__ == "__main__":
    asyncio.run(main())

##python script.py --input_path /path/to/requests.csv --save_path /path/to/output.csv --api_key YOUR_API_KEY
