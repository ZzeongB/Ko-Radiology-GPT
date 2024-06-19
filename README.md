# 🚑🏥☢️ Ko-Radiology-GPT ☢️🏥🚑
### Real-time Q&A Large Language Model focusing on chest X-ray radiology report **in Korean**
![Demo Image](demo.png)
## 💡 Introduction
Ko-Radiology-GPT는 한국어로 작성된 흉부 X-선 방사선 보고서에 초점을 맞춘 실시간 질의응답 대형 언어 모델입니다. 이는 기존에 영어 기반의 방사선 판독보고서 챗봇 [Hippo](https://github.com/healthhub-ai/SNU-Radiology-GPT/)를 한국어로 확장하고 개선한 것으로, 복잡한 의료 용어와 방사선 판독 결과를 이해하고, 사용자의 질문에 대해 정확하고 신속하게 답변할 수 있습니다.

* Base Model: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  
* Dataset: 
    - (translated) [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/): 약 160k개의 노트를 Google Translate을 통해 번역하여 사용하였습니다.  
    - [의료, 법률 전문 서적 말뭉치 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71487)
    - [전문분야 한영 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=111)
* Method: Instruction-following(by Stanford Alpaca) 방식으로 학습을 진행하였습니다. 데이터의 생성에는 GPT-3.5 turbo API를 이용하였습니다.


## 💻 Environment
제공드린 Dockerfile을 사용하시면 됩니다.  

#### 1. Docker Image Build
```bash
docker build -t hippo:latest .
```

#### 2. Docker Run Container
```bash
docker run -v MOUNT_PATH:/workspace --gpus GPU_NUM -it --name "hippo" hippo:latest
```
-v 옵션을 지정하여 볼륨을 마운트하였습니다. MOUNT_PATH는 컨테이너에 마운트할 로컬 경로를 의미합니다.  
--gpus 옵션을 지정하여 사용할 GPU를 지정할 수 있습니다.  
-it 옵션을 지정하여 터미널을 이용하여 컨테이너와 상호작용할 수 있습니다.  
"hippo"는 컨테이너의 이름, hippo:latest는 이미지 이름입니다.  

3. Container 재사용  
실행중인 컨테이너에 재진입하여 작업하는 경우, 다음의 명령어를 사용하시면 됩니다.
```bash
docker exec -it hippo /bin/bash
```
hippo는 실행중인 컨테이너의 이름입니다.

### Requirements
이후 필요한 모듈들을 아래 명령어로 다운로드하면 됩니다.
```bash
bash setup.sh
```

## 📚 Data Preprocessing
#### 1. MIMIC-CXR 한국어 번역

[MIMIC-CXR 사이트](https://physionet.org/content/mimic-cxr/2.0.0/)를 통해 데이터를 다운받은 후, 아래 명령어를 통해 한국어로 번역해보세요.
```bash
python translate.py --input-path INPUT_PATH --output-path OUTPUT_PATH
```

#### 2. Data Preprocessing

번역된 MIMIC-CXR 데이터 및 AI hub 데이터들의 전처리 과정입니다. 
`{id, note}` 형식의 csv 데이터를 `input/` 디렉토리에 저장하고, 아래 명령어를 실행하면, `output/` 디렉토리에 결과가 저장됩니다. `API_KEY` 자리에는 OpenAI에서 발급받은 API Key를 입력하면 됩니다. 각 과정의 디테일은 아래 Details of User Manual을 참고해주세요!

```bash
bash preprocess.sh API_KEY
```  

## 🔄 Fine-tuning Llama Model
이 프로젝트는 특정 작업을 위해 Llama 모델을 파인 튜닝하는 것을 포함합니다. 모델은 사용자 정의 데이터셋에서 훈련되며, 훈련 과정은 다양한 매개변수로 사용자 정의할 수 있습니다.
먼저, 아래 명령어로 Huggingface-Cli에 로그인해야 합니다.
```bash
# huggingface에 로그인하신 후 토큰을 발급하세요. Y/n 질문에는 n 으로 대답하면 됩니다.
huggingface-cli login
```

아래 명령어를 실행하여 fine-tuning으르 하면 됩니다. 
```bash
python "src/fine_tuning.py" \
--output_dir OUTPUT-DIR \
--model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
--data_path DATA-PATH \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-4 \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing True \
--ddp_timeout 1800
```
* OUTPUT-DIR: 학습된 모델이 저장될 경로입니다. 
* DATA-PAT: 전처리된 데이터셋이 저장되어 있는 경로입니다.
다른 파라미터들도 필요에 따라 수정할 수 있습니다.

아래 명령어는 예시입니다.
```bash
# Example: with 006-Medical dataset
python "src/fine_tuning.py" \
--output_dir "./finetuned_model" \
--model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
--data_path "output/006-Medical_postprocess.jsonl" \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-4 \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing True \
--ddp_timeout 1800
```

## 🤖 Inference
학습된 모델을 이용하여 답변을 생성하고자 하는 경우, 다음의 명령어를 사용하시면 됩니다.  

```bash
python src/inference.py --ft_path MODEL_PATH
```
* MODEL_PATH: 학습된 모델이 존재하는 repository를 입력하세요.
해당 모듈에서는 학습된 radiology_GPT가 챗봇 형식으로 사용자와 질의응답을 하게 됩니다. 이전에 이루어졌던 대화를 반영하여 답변을 생성하게 됩니다.  
* PreTrained 모델은 `h2a0e0u2n/changtongsul`에서 찾아볼 수 있습니다. 아래 명령어로 pretrained model을 이용해보세요!

```bash
# Example: pre-trained model
python src/inference.py --ft_path h2a0e0u2n/changtongsul
```

## 🌐 Demo
아래 모듈에서는 사용자가 채팅 인터페이스를 통해 Ko-Radiology-GPT를 사용해 볼 수 있습니다.  
Demo는 Gradio 라이브러리를 사용하여 구현되었습니다. (Huggingface chat demo 코드 참조)  
명령어 실행 후 터미널에 출력되는 public url을 클릭하시면 Demo를 사용하실 수 있습니다.

```bash
python src/app.py --model MODEL_PATH
```

* MODEL_PATH: 학습된 모델이 존재하는 repository를 입력하세요. 
* Pretrained된 모델을 이용하고 싶다면 `ko-gpt`를 입력하세요. default 값은 기존 llama-2-7b-chat 모델입니다.

```bash
# Example: pre-trained model
python src/app.py --model ko-gpt
```


## 📊 Compare and Evaluate
### Comparison
다른 모델들의 답변을 받아 보고 싶으실 경우, Comparison 디렉토리에 있는 모듈들을 활용하시면 됩니다.  

#### 1. llama2.py
Fine-tuning 전 Llama2 model의 답변을 받아오고 싶을 때 사용하시면 됩니다. 라마2에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'llama2_answer'이라는 새로운 column에 답변을 저장하여 명시해주신 save_path에 csv 파일로 반환합니다.

Llama2를 사용하기 위해서는 huggingface CLI login이 필요합니다. 앞서 Fine Tuning 섹션에서 설명드린 방법대로 CLI login을 진행해주시면 됩니다.
```bash
python comparison/llma2.py --input_path INPUT_PATH --save_path OUTPUT_PATH
```
* INPUT_PATH: path to csv input file
* OUTPUT_PATH: path to csv output file

#### 2. hippo.py
이전 프로젝트에서 개발한 영어 질의응답 모델인 hippo의 답변을 받아오고 싶을 때 사용하시면 됩니다. Hippo에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'hippo_answer'이라는 새로운 column에 답변을 저장하여 명시해주신 save_path에 csv 파일로 반환합니다.
Base model을 Llama2로 하고 있기에, 역시 huggingface CLI login이 필요합니다. 
```bash
python comparison/hippo.py --input_path INPUT_PATH --save_path OUTPUT_PATH --hippo_model_dir MODEL_PATH
```
* INPUT_PATH: path to csv input file
* OUTPUT_PATH: path to csv output file
* MODEL_PATH: path to pretrained model 

#### 3. koGPT.py
이번 프로젝트에서 개발한 한국어 질의응답 모델인 Ko-Radiology-GPT의 답변을 받아오고 싶을 때 사용하시면 됩니다. Ko-Radiology-GPT에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'koGPT_answer'이라는 새로운 column에 답변을 저장하여 명시해주신 save_path에 csv 파일로 반환합니다.
Base model을 Llama2로 하고 있기에, 역시 huggingface CLI login이 필요합니다. 
```bash
python comparison/koGPT.py --input_path INPUT_PATH --save_path OUTPUT_PATH --fine_tuned_model_dir MODEL_PATH
```

* INPUT_PATH: path to csv input file
* OUTPUT_PATH: path to csv output file
* MODEL_PATH: path to pretrained model 

### Evaluation
GPT-4를 이용하여 평가가 이루어집니다. 평가 지표는 1. Accuracy 2. Conciseness 3. Consistency 4. Understandability의 4가지가 있으며, 각 1점부터 4점까지 점수를 매겨 이루어집니다. 더불어 Ko-GPT와 다른 영어 모델의 답변 유사도를 보기 위하여 5. Similarity를 추가로 측정하였습니다.
GPT-4로부터 각 모델의 답변을 평가하는 경우, `evaluate.py`를 이용하면 됩니다.

아래의 모든 모듈을 사용하기 위해서는 secrets.py 파일이 필요합니다. 해당 파일은 다음과 같은 내용을 담고 있으며, evaluate_*.py 파일들과 동일한 directory hierarchy에 위치시켜두시면 됩니다.

```python
OPENAI_API_KEY = "your API key"  ## GPT4를 사용하기 위한 OpenAI API key
```

이후 아래 명령어를 사용하여 Evaluation을 진행할 수 있습니다.
```bash
python ./evaluate.py --input_path INPUT_PATH --output_path OUTPUT_PATH --type TYPE 
```
* INPUT_PATH: path to csv input file
* OUTPUT_PATH: path to csv output file
* TYPE: acc(accuracy), coc(conciseness), cos(consistency), und(underestandability), sim(similarity) 중 하나를 입력하면 됩니다.

## [Appendix] Details of User Manual
###  Data Preprocessing

Data Generation은 다음의 단계를 거쳐 이루어집니다.  

#### 0. (MIMIC-CXR data only) MIMIC-CXR 데이터셋에서 방사선 판독보고서 파일인 notes를 전처리합니다.  

보고서마다 형식이 제각각이기 때문에, 보고서에서 핵심 정보를 담고 있는  **"EXAMINATION", "HISTORY", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", "IMPRESSION"**  항목을 중심으로 전처리를 수행하였습니다.

```bash
python preprocessing/preprocess_mimic_cxr.py --input_path INPUT_PATH --save_path SAVE_PATH
```
* INPUT_PATH: MIMIC-CXR notes 데이터셋이 위치한 경로입니다.  
* SAVE_PATH: 전처리된 데이터셋이 저장될 경로입니다.


#### 1. Instruction generator
OpenAI API를 이용하여 instruction을 생성합니다.  
```bash
python preprocessing/instruction_generator.py --input_path INPUT_PATH --save_path SAVE_PATH --api_key API_KEY
```  
이 때 max_requesets/token_per_minute, max_attemps 등 API 세부 설정을 변경하실 수 있습니다.  
세부 파라미터는 코드를 참조하세요!  
  
2. API Response에서 생성된 Instruction을 후처리하여, 각 Instruction에 대한 answer를 생성하도록 명령하는 prompt를 생성합니다.
```bash
python preprocessing/postproc_question.py --input_path INPUT_PATH --save_path SAVE_PATH
```  
  
#### 3. Answer generator
OpenAI API를 다시 이용하여 후처리한 데이터에 대한 Answer를 생성합니다.  
```bash
python preprocessing/answer_generator.py --input_path INPUT_PATH --save_path SAVE_PATH --api_key API_KEY
```  
#### 4. Instruction-Answer postprocess 
생성한 Instruction-Answer 쌍을 후처리합니다.  
```bash
python preprocessing/answer_postprocess.py --input_path INPUT_PATH --save_path SAVE_PATH
```

#### 5. Convert to fine-tuning version
마지막으로 후처리된 데이터를 fine-tuning을 위한 형식으로 고칩니다.
```bash
python preprocessing/csv_to_jsonl_converter.py --input_path INPUT_PATH --save_path SAVE_PATH
```
자, 이제 `SAVE_PATH`에는 fine-tuning을 위해 필요한 데이터셋이 담겨있습니다.

## Reference
[KAIST Asclepius](https://github.com/starmpcc/Asclepius)  
[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  
[Open AI](https://github.com/openai/openai-cookbook/tree/main)  
[Huggingface Llama2 chat demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py)  