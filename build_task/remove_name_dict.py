import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove name_dict from GraphTextEncoder.
    # Drop self.name_dict from __init__.
    content = re.sub(
        r'self\.name_dict = \{node: int\(index\+1\) for index, node in enumerate\(sorted\(graph\.nodes\(\)\)\)\}',
        '',
        content
    )
    content = re.sub(
        r'self\.name_dict = \{node: int\(node\) for node in sorted\(graph\.nodes\(\)\)\}',
        '',
        content
    )
    content = re.sub(
        r'self\.preserve_original_ids = preserve_original_ids\n\s+if preserve_original_ids:\n\s+',
        '',
        content
    )
    content = re.sub(
        r'self\.preserve_original_ids = preserve_original_ids\n',
        '',
        content
    )

    # 2. Replace name_dict.values() in node_encoder.
    content = re.sub(
        r'node_list = sorted\(self\.name_dict\.values\(\)\)',
        'node_list = sorted(self.graph.nodes())',
        content
    )

    # 3. Replace name_dict[u] and name_dict[v] in edge_encoder.
    content = re.sub(
        r'edge_list = \[f"\(\{self\.name_dict\[u\]\}, \{self\.name_dict\[v\]\}, \{w\[weight_key\]\}\)" for u, v, w in edges\]',
        'edge_list = [f"({u}, {v}, {w[weight_key]})" for u, v, w in edges]',
        content
    )
    content = re.sub(
        r'edge_list = \[f"\(\{self\.name_dict\[u\]\}, \{self\.name_dict\[v\]\}\)" for u, v in self\.graph\.edges\(\)\]',
        'edge_list = [f"({u}, {v})" for u, v in self.graph.edges()]',
        content
    )

    # 4. Replace name_dict[x] -> x in other files.
    content = re.sub(
        r'name_dict\[([^\]]+)\]',
        r'\1',
        content
    )

    # 5. Replace int(name_dict[u]) -> int(u).
    content = re.sub(
        r'int\(name_dict\[([^\]]+)\]\)',
        r'int(\1)',
        content
    )

    # 6. Replace tuple(int(name_dict[v]) for v in tup) -> tuple(int(v) for v in tup).
    content = re.sub(
        r'tuple\(int\(name_dict\[([^\]]+)\]\) for \1 in tup\)',
        r'tuple(int(\1) for \1 in tup)',
        content
    )
    # Use a more general replacement.
    content = re.sub(
        r'tuple\(int\(name_dict\[([^\]]+)\]\) for ([^)]+) in tup\)',
        r'tuple(int(\2) for \2 in tup)',
        content
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Processed: {filepath}")

files = [
    r'xxx',
    r'xxx',
    r'xxx',
]

for f in files:
    process_file(f)
