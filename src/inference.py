# 모델을 테스트해보세요. 답변이 만족스럽지 않을 경우 1번 더 실행해보세요.
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# model_path에 본인의 final_output_2가 저장된 경로를 입력해주세요
model_path = "/content/drive/My Drive/changtongsul/final_output_2"

# Load the tokenizer and model from the path where they were saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoPeftModelForCausalLM.from_pretrained(model_path)

def ask_question(question, model, tokenizer):
    """Function to ask a question to the model and get an answer."""
    # Encode the question to model input
    input_ids = tokenizer.encode(question, return_tensors="pt")

    # Generate a response from the model // 답변이 길어서 문장 중간에 잘릴 경우 max_length를 늘려주세요.
    output_ids = model.generate(input_ids, max_length=1000)

    # Decode the output ids to a string
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# 보고서 : ... 질문 : ... 형식으로 원하는 보고서와 질문을 입력해주세요.
question = "보고서: 병력: 폐렴을 평가하기 위해 왼쪽 하엽과 상엽의 호흡음이 감소된 발열. 조사 결과: 연구와 비교하여 여러 척추체에 압박이 나타나며 후만증이 증가할 수 있습니다. 그러나 급성 국소 폐렴, 혈관 울혈 또는 흉막 삼출의 증거는 없습니다. 승모판 고리의 석회화가 다시 기록됩니다. 질문: 주어진 보고서에 따르면 이 환자의 척추체 상태는 어떠한가요?"

# Get the answer from the model
answer = ask_question(question, model, tokenizer)

# Print the answer
print(answer)