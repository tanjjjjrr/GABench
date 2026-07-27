import re

with open(r'xxx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace name_dict[...] -> ...
# This regex matches name_dict[xxx] expressions.
content = re.sub(r'name_dict\[([^\]]+)\]', r'\1', content)

# Replace int(name_dict[u]) -> int(u)
content = re.sub(r'int\(name_dict\[([^\]]+)\]\)', r'int(\1)', content)

# Replace tuple(int(name_dict[v]) for v in tup) -> tuple(int(v) for v in tup)
content = re.sub(r'tuple\(int\(name_dict\[([^\]]+)\]\) for ([^)]+) in tup\)', r'tuple(int(\2) for \2 in tup)', content)

with open(r'xxx', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
