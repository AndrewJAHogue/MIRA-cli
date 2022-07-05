import json

def get_computer_path():
    path = ''
    with open('./paths.json', 'r+') as f:
        f_data = json.load(f)
        path = f_data['computer path']
        
    return path

