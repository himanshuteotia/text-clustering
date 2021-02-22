
def split_list(arr, n):
    l = len(arr)
    return [arr[i * l // n: (i+1)*l//n] for i in range(n)]

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

