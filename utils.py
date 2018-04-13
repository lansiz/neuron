import pickle 

def read_pickle(file_name):
    try:
        f = open('./' + file_name, 'rb')
        o = pickle.load(f)
        f.close()
        return o
    except:
        return None


def write_pickle(o, file_name):
    try:
        f = open('./' + file_name, 'wb')
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return True
    except:
        return False

