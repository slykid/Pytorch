# 1. 정규표현식 (Regular Expression)
import re

# 전화번호 검색에 대한 정규표현식
regex = r"([\w]+\s*:?\s*)?\(?\+?([0-9]{1,3})?\-?[0-9]{2,3}(\)|\-)?[0-9]{3,4}\-?[0-9]{4}"
x = "Ki: +82-10-9420-1356"
re.sub(regex, "REMOVED",x)

x = "CONTENT jiu 02)9420-1246"
re.sub(regex, "REMOVED", x)

# 한국어 분절 : MeCab
import MeCab

m = MeCab.Tagger()
out = m.parse('안녕하세요, 반갑습니다!')
print(out)
