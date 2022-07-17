'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import re
import sys
import time
import math
from typing import T_co

import pandas as pd
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class ImageNetteDataset(Dataset):
    def __init__(self, path, labels_file, transforms=None):
        self.path = path
        self.transforms = transforms
        category_mapping = pd.read_csv(labels_file, sep=' ', names=['label', 'category']).reset_index()
        category_mapping['label'] = category_mapping['label'].sort_values().values
        categories = os.listdir(path)
        self.file_labels = pd.DataFrame(columns=['file', 'label'])
        concats = []
        for c in tqdm(categories):
            files = os.listdir(os.path.join(path, c))
            for f in files:
                id = int(category_mapping[category_mapping.loc[:, 'label'] == c]['index'])
                # label = category_mapping[category_mapping.loc[:, 'label'] == c]['category'].replace(' ', '_')
                concats.append(pd.Series({'file': os.path.join(c, f), 'id': id,  'label': c}))
        self.file_labels = pd.DataFrame(concats)

    def __len__(self):
        return len(self.file_labels)

    def __getitem__(self, index) -> T_co:
        p = self.file_labels.iloc[index, 0]
        image = read_image(os.path.join(self.path, p))
        if image.shape[0] != 3:
            raise ValueError(f'Image must have 3 dimensions, found {image.shape[0]}')
        image = self.transforms(image / image.max())
        cat_id = self.file_labels.iloc[index, 1]

        return image, cat_id