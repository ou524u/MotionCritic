from render.render import render_multi
import os
import torch
import argparse
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
# Initialize argparse
parser = argparse.ArgumentParser(description='myparser')

parser.add_argument('--folder', type=str, default=None,
                    help='Path to the saved model checkpoint to load (default: None)')

parser.add_argument('--gpu_index', type=int, default=None,
                    help='Index of the GPU to use (default: 0)')

parser.add_argument('--exclude_gt', action='store_true',
                    help='Initiate a bigger model')

parser.add_argument('--no_comment', action='store_true',
                    help='Initiate a bigger model')


args = parser.parse_args()

render_gt = True
if args.exclude_gt:
    render_gt = False

no_comment = args.no_comment


if args.gpu_index is not None:
    gpu_index = args.gpu_index
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_index}'
    device = torch.device("cuda")

else:
    device = torch.device("cpu")



def file_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    files = os.listdir(folder_path)
    print(f"checking file {files}")
    
    pth_files = [file for file in files if file.endswith('.pth')]
    
    return pth_files

def render_file(filename, no_comment=False):
    data = torch.load(filename, map_location=device)

    check_list = data['motion']
    comment_list = data['comment']
    path_list = data['path']
    
    render_multi(check_list.to(device), device, comment_list, path_list, no_comment=no_comment)




files = file_from_folder(os.path.join(PROJ_DIR, args.folder))
print(f"files are {files}, device is {device}")
for file in files:
    render_file(os.path.join(os.path.join(PROJ_DIR, args.folder), file), no_comment=no_comment)