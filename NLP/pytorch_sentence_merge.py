import sys, fileinput
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    # 입력 문장
# 자연어처리는 인공지능의 한 분야이다. 시퀀스 투 시퀀스의 등장 이후
# 딥러닝을 활용한 자연어처리는 새로운 시대를 맞이했다. 문장을 입력으로
# 해서 단순 수치로 표현하는 것이 아닌 원하는 데로 문장을 생성하는 것도
# 가능하다.

    buf = []

    for line in fileinput.input():
        if line.strip() != '':
            buf += [line.strip()]
            sentences = sent_tokenize(" ".join(buf))

            if len(sentences) > 1:
                buf = sentences[-1:]

                sys.stdout.write('\n'.join(sentences[:-1]) + '\n')
    sys.stdout.write(' '.join(buf) + '\n')