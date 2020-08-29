import os
import numpy as np
from sklearn import preprocessing
from audio import power2db, log_power_normalize
from sklearn.externals import joblib

#train/test每一行都只是一个文件名
TRAIN_FILE = 'LJSpeech-1.1/train.txt'
TEST_FILE = 'LJSpeech-1.1/test.txt'
Linears_DIR = 'LJSpeech-1.1/linear_from_generate_batch'
PPGs_DIR = 'LJSpeech-1.1/ppg_from_generate_batch'
# Linear_DIM = 201
# PPG_DIM = 345

outdir = 'LJSpeech-1.1_Norm_Sort'
Norm_PPGs_DIR = os.path.join(outdir, 'norm_ppg')
Norm_DB_Linears_DIR = os.path.join(outdir, 'norm_db_linear')

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(Norm_PPGs_DIR):
    os.makedirs(Norm_PPGs_DIR)

if not os.path.exists(Norm_DB_Linears_DIR):
    os.makedirs(Norm_DB_Linears_DIR)

def text2list(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.split()[0])
    return file_list

def spec2normDB_inSignal(spec):
    spec = spec ** 2
    db_spec = power2db(spec)
    norm_db = log_power_normalize(db_spec)
    return norm_db

# ppg is predicted vector
def get_single_data_pair(fname, ppgs_dir, linears_dir):
    ppg_f = os.path.join(ppgs_dir, fname+'.npy')
    linear_f = os.path.join(linears_dir, fname+'.npy')
    ppg = np.load(ppg_f)
    linear = np.load(linear_f)
    assert ppg.shape[0] == linear.shape[0]
    return ppg, linear, ppg.shape[0]


def main():
    # standard_scalar = preprocessing.StandardScaler()
    # linear_standard_scalar = preprocessing.StandardScaler()
    sorted_train_list = []
    sorted_test_list = []

    train_list = text2list(TRAIN_FILE)
    test_list = text2list(TEST_FILE)

    for x in train_list:
        ppg, linear, len = get_single_data_pair(x, PPGs_DIR, Linears_DIR)
        sorted_train_list.append((x, len))
        # standard_scalar.fit(ppg)
        # linear_standard_scalar.fit(spec2normDB_inSignal(linear))
        # break
    for x in test_list:
        ppg, linear, len = get_single_data_pair(x, PPGs_DIR, Linears_DIR)
        sorted_test_list.append((x, len))
        # standard_scalar.fit(ppg)
        # linear_standard_scalar.fit(spec2normDB_inSignal(linear))
        # break


    sorted_train_list.sort(key=lambda s: s[1])
    sorted_test_list.sort(key=lambda s: s[1])
    with open(os.path.join(outdir, 'sorted_train.txt'), 'w') as f:
        for x in sorted_train_list:
            f.writelines(x[0] + '\n')
    with open(os.path.join(outdir, 'sorted_test.txt'), 'w') as f:
        for x in sorted_test_list:
            f.writelines(x[0] + '\n')

    for x in sorted_train_list:
        ppg, linear, len = get_single_data_pair(x[0], PPGs_DIR, Linears_DIR)
        # norm_ppg = standard_scalar.transform(ppg)
        norm_ppg = ppg
        np.save(os.path.join(Norm_PPGs_DIR, x[0]), norm_ppg)
        # norm_normDB_linear = linear_standard_scalar.transform(spec2normDB_inSignal(linear))
        norm_normDB_linear = spec2normDB_inSignal(linear)
        np.save(os.path.join(Norm_DB_Linears_DIR, x[0]), norm_normDB_linear)
    for x in sorted_test_list:
        ppg, linear, len = get_single_data_pair(x[0], PPGs_DIR, Linears_DIR)
        # norm_ppg = standard_scalar.transform(ppg)
        norm_ppg = ppg
        np.save(os.path.join(Norm_PPGs_DIR, x[0]), norm_ppg)
        # norm_normDB_linear = linear_standard_scalar.transform(spec2normDB_inSignal(linear))
        norm_normDB_linear = spec2normDB_inSignal(linear)
        np.save(os.path.join(Norm_DB_Linears_DIR, x[0]), norm_normDB_linear)

    # joblib.dump(standard_scalar, 'standard_scalar')
    # joblib.dump(linear_standard_scalar, 'linear_standard_scalar')


if __name__ == '__main__':
    main()
