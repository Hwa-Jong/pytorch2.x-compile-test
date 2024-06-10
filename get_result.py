import os
import re
import numpy as np

import matplotlib.pyplot as plt


def main():
    path = 'results'
    
    names = os.listdir(path)
    txt_list = []
    png_list = []
    npy_list = []
    data = {}
    for name in names:
        base, ext = os.path.splitext(name)
        if ext == '.png':
            png_list.append(os.path.join(path, name))
        elif ext == '.npy':
            npy_list.append(os.path.join(path, name))
        elif ext == '.txt':
            txt_list.append(os.path.join(path, name))
            
    for name in txt_list:
        now_data = load_txt_data(name)
        data[name] = now_data
        
    report = get_report(data, save_graph=True)
    
    print('< Results >')
    for k, v in report.items():
        print(k)
        print(f'Median time : {v[0]} | Total time : {v[1]}| Speed up rate : {v[3]}| Accuracy : {v[2]}\n')
    
    print('finish')
    
def load_npy_data(path):
    data = np.load(path)
    return data

def load_txt_data(path):
    data = []
    result = {}
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line in ['', None]:
                data.append(line)
                epoch, acc, time = split_data_in_txt(line)
                               
                if time is None:
                    result.setdefault(epoch, []).append(acc)
                if acc is None:
                    result.setdefault(epoch, []).append(time)
                    
    return result
   
def split_data_in_txt(txt):
    epoch = None
    acc = None
    time = None
    data = txt.split(' ')
    if data[0].lower() == 'epoch':
        num = re.sub(r'[^a-zA-Z0-9\s]', '', data[1])
        epoch = f'epoch_{num}'
        if len(data) > 5:
            acc = float(re.sub(r'[^a-zA-Z0-9\s.]', '', data[-1]))
        else:
            time = float(re.sub(r'[^0-9\s.]', '', data[-1]))
        print()
    else:
        epoch = data[0].lower()
        h, m, s = float(re.sub(r'[^0-9\s.]', '', data[-3])), float(re.sub(r'[^0-9\s.]', '', data[-2])), float(re.sub(r'[^0-9\s.]', '', data[-1]))
        time = h * 3600 + m * 60 + s
        
    return epoch, acc, time

def get_report(data:dict, save_graph):    
    report = {}
    if save_graph:
        plt.figure(figsize=(10, 6))
        plt.title('Time per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time(s)')
        
    total_times = []
    for k, v in data.items():
        acc_list = []
        time_list = []
        for _k, _v in v.items():
            if _k.lower() != 'total':
                acc_list.append(_v[0])
                time_list.append(_v[1])
                
            else:
                total_time = _v[0]
                
        acc_list = np.array(acc_list)
        time_list = np.array(time_list)
        key = os.path.splitext(os.path.basename(k))[0]
        if save_graph:
            plt.plot(range(1, 21), time_list, label=key)
    
        median_time = np.median(time_list)
        report[key] = [median_time, total_time, acc_list.max()]
        total_times.append(median_time)
        
    # add speed up rate
    max_time = max(total_times)
    for k, v in report.items():
        v.append(np.round(max_time/v[0], 3))
        
    if save_graph:
        plt.xticks(range(1, len(time_list)+1))
        plt.grid(True)
        plt.legend()
        plt.savefig('results\\result.png')
    return report
    
    
    
if __name__ == '__main__':
    main()