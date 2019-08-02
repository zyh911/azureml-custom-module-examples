from .imagenet1000_index_to_label import my_dict
new_dict = {}
for k, v in my_dict.items():
    lists = v.split(',')
    for i in lists:
        i = i.strip()
        new_dict[i] = k
print(new_dict)
with open('imagenet1000_label_to_index.py', 'w') as f:
    f.write('new_dict = {\n')
    for k, v in new_dict.items():
        f.write('    "' + k + '": ' + str(v) + ',\n')
    f.write('}')
