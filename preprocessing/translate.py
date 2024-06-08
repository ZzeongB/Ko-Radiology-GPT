import pandas as pd
import time
import argparse
from googletrans import Translator

google = Translator()

def preprocess_note(notes):
    res = ""
    prev = ""

    notes = notes.splitlines()

    for line in notes:
        if line and line[0] == " ":
            if len(line) == 1:
                res += prev
                prev = ""
            elif line.find(":") != -1:
                prev += "\n\n"
                prev += line.strip("\n")
            else:
                prev += line.strip("\n")
    res += prev

    return res

def main(input_path, output_path):
    df_origin = pd.read_csv(input_path, names=['', 'id', 'note'])

    df_result = pd.DataFrame(columns=['id', 'note', 'note_kor'])
    i = 0

    for note_lines in df_origin["note"]:
        id = df_origin['id'].iloc[i]
        res = preprocess_note(note_lines)

        note_lines_kor = google.translate(res, dest="ko").text
        df_result.loc[i] = [id, res, note_lines_kor]

        i += 1
        if i % 100 == 0:
            time.sleep(2)

        if i % 1000 == 0:
            time.sleep(10)
            df_result.to_csv(output_path, mode='a', header=False, encoding="utf-8-sig")
            df_result = pd.DataFrame(columns=['id', 'note', 'note_kor'])
            print(f"{i} files have been translated")

    df_result.to_csv(output_path, mode='a', header=False, encoding="utf-8-sig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate notes.')
    parser.add_argument('--input-path', type=str, required=True, help='Input file path')
    parser.add_argument('--output-path', type=str, required=True, help='Output file path')

    args = parser.parse_args()

    main(args.input_path, args.output_path)