#!/bin/bash

INPUT_PATH="/home/airdk/jeongin/Ko-GPT/input/"
SAVE_PATH="/home/airdk/jeongin/Ko-GPT/output"
API_KEY=$1

for FILE_PATH in $INPUT_PATH*.csv
do
    FILE=$(basename $FILE_PATH .csv)
    python3 preprocessing/instruction_generator.py --input_path $FILE_PATH --save_path $SAVE_PATH/${FILE}_instruction.jsonl --api_key $API_KEY
    python3 preprocessing/postproc_question.py --input_path $SAVE_PATH/${FILE}_instruction.jsonl --save_path $SAVE_PATH/${FILE}_question.jsonl
    python3 preprocessing/answer_generator.py --input_path $SAVE_PATH/${FILE}_question.jsonl --save_path $SAVE_PATH/${FILE}_answer.jsonl --api_key $API_KEY
    python3 preprocessing/answer_postprocess.py --input_path $SAVE_PATH/${FILE}_answer.jsonl --save_path $SAVE_PATH/${FILE}_postprocess.jsonl
    python preprocessing/csv_to_jsonl_converter.py --input_path $SAVE_PATH/${FILE}_postprocess.jsonl --save_path $SAVE_PATH/${FILE}_final.jsonl

done