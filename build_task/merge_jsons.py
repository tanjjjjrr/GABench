from pathlib import Path
import json

BASE = Path('task_output1')

DATASETS = {
    'graph_info_retrieval': 'graph_info_retrieval',
    'graph_theory': 'graph_theory',
}

TYPES = {
    'node': 'node',
    'edge': 'edge',
    'graph': 'graph',
}

def collapse_array(json_str, field_name):
    """Collapse an expanded array field into a single line."""
    start_pat = f'"{field_name}": ['
    pos = 0
    result = ''
    while True:
        idx = json_str.find(start_pat, pos)
        if idx == -1:
            result += json_str[pos:]
            break
        result += json_str[pos:idx]
        bracket_start = idx + len(start_pat)
        depth = 1
        i = bracket_start
        while i < len(json_str) and depth > 0:
            if json_str[i] == '[':
                depth += 1
            elif json_str[i] == ']':
                depth -= 1
            i += 1
        raw_content = json_str[bracket_start:i-1]
        collapsed = ''.join(raw_content.split()).replace('][', '],[')
        result += start_pat + collapsed + ']'
        pos = i
    return result

def quote(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

output_path = BASE / 'merged.json'
with open(output_path, 'w', encoding='utf-8') as out:
    out.write('{\n')
    datasets = list(DATASETS.items())
    for i, (src_key, dataset_name) in enumerate(datasets):
        out.write('  ' + quote(dataset_name) + ': {\n')
        types_list = list(TYPES.items())
        for j, (type_key, type_name) in enumerate(types_list):
            json_file = BASE / src_key / f'{type_key}.json'
            if not json_file.exists():
                continue
            with open(json_file, 'r', encoding='utf-8') as f:
                json_str = f.read()
            # Graph-level records need compact nodes and edges arrays.
            if type_key == 'graph':
                json_str = collapse_array(json_str, 'nodes')
                json_str = collapse_array(json_str, 'edges')
            records = json.loads(json_str)
            print(f'  {dataset_name} - {type_name}: {len(records)} records')
            out.write('    ' + quote(type_name) + ': [\n')
            # Extract each original JSON object while preserving compact formatting.
            pos = 0
            # Skip the opening '['.
            json_str = json_str.strip()
            if json_str.startswith('['):
                pos = 1
            # Skip whitespace.
            while pos < len(json_str) and json_str[pos] in ' \t\n\r':
                pos += 1
            for k, rec in enumerate(records):
                if json_str[pos] == ',':
                    pos += 1
                while pos < len(json_str) and json_str[pos] in ' \t\n\r':
                    pos += 1
                # Find the end of this object.
                depth = 0
                start = pos
                while pos < len(json_str):
                    c = json_str[pos]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                    elif c == '"':
                        pos += 1
                        while pos < len(json_str):
                            if json_str[pos] == '\\':
                                pos += 2
                            elif json_str[pos] == '"':
                                pos += 1
                                break
                            else:
                                pos += 1
                    else:
                        pos += 1
                rec_str = json_str[start:pos].strip()
                out.write('      ' + rec_str + ',\n')
            out.write('    ]')
            out.write(',\n' if j < len(types_list) - 1 else '\n')
        out.write('  }')
        out.write(',\n' if i < len(datasets) - 1 else '\n')
    out.write('}\n')

print(f'\nMerge complete -> {output_path}')
