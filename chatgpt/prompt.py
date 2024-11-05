import os.path

import pandas as pd
import re
from openai import OpenAI
import time

key = 'omitted'
client = OpenAI(api_key=key)
def query_chatgpt():
    min_cost = 3
    df = pd.read_csv('../data/test.csv')
    print(df.shape)
    with open('chatgpt.out', 'r') as file:
        lines = file.readlines()
    cursor = len(lines)

    with open('chatgpt.out', 'a') as ret:
        for (i, item) in df.iterrows():
            if i <  cursor:
                continue
            print('=' * 50, i, '=' * 50)
            func = item['function']
            prompt = "Given the vulnerable function, briefly suggest how to fix it within 100 words.\n" + func
            while True:
                try:
                    completion = client.chat.completions.create(
                        temperature=0,
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    # print(completion.choices[0].message)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(60)

            reason = completion.choices[0].finish_reason
            if reason == 'stop':
                content = completion.choices[0].message.content
                print(content)
                ret.write(str(i)+'\t'+content.replace('\n', '')+'\n')
            else:
                print(reason)
                print(completion.choices[0].message.content)
            time.sleep(min_cost)

    # df.to_csv('')



if __name__ == '__main__':
    token_num = 0
    query_chatgpt()
