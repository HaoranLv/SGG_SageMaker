import h5py
import json
with open('/Users/lvhaoran/gitworkplace/SGG/datasets/vg/VG-SGG-dicts-with-attri.json','r') as ljs:
    dic=json.load(ljs)
    print(len(dic))
    for key,value in dic.items():
        print(key)
    # print(dic['idx_to_attribute'])
with h5py.File('/Users/lvhaoran/gitworkplace/SGG/datasets/vg/VG-SGG-with-attri.h5',"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
        print(f[key][-1])
