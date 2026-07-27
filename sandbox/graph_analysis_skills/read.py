import torch
d = torch.load("xxx", weights_only=False)
print(type(d), len(d))
item = d[0]
print(type(item))
if isinstance(item, dict):
    print("keys:", list(item.keys()))
else:
    print("attrs:", [a for a in dir(item) if not a.startswith('_')])
