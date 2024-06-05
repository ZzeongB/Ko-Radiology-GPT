import pandas as pd
import json
import os
import argparse

# argparse 객체 생성
parser = argparse.ArgumentParser()

# 입력받을 인자 등록
parser.add_argument('--input_path', required=True, help='Input file path')
parser.add_argument('--save_path', required=True, help='Output file path')

# 입력받은 인자를 args에 저장
args = parser.parse_args()

# pandas로 csv 파일 읽기
inputs = args.input_path.split(',')

for input in inputs:
    data = pd.read_csv(input, encoding="utf-8")

    data = data.rename(columns={"Unnamed: 0": "id"})

    # 각 행을 별도의 JSON 객체로 변환하고, 이를 줄 단위로 파일에 쓰기
    # replace input path with save path
    output = input.replace(args.input_path, args.save_path)
    with open(output, 'w', encoding='utf-8') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file, ensure_ascii=False)
            file.write('\n')