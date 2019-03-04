import re

with open("differential_result.txt","r") as f:
    content = f.read()

pattern = re.compile("\d+\.\d+")
result = sorted(list(map(float,re.findall(pattern,content))),reverse = True)
print(result)    