import re

with open("lung.txt","r") as f:
    content = f.read()

pattern = re.compile("\d+\.\d+")
result = max(map(float,re.findall(pattern,content)))
print(result)    