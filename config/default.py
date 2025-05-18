class CFG:
    eng2de = {
        'train': '/content/drive/MyDrive/wmt-2014-en2de/wmt14_translate_de-en_train.csv',
        'test': '/content/drive/MyDrive/wmt-2014-en2de/wmt14_translate_de-en_test.csv',
        'val': '/content/drive/MyDrive/wmt-2014-en2de/wmt14_translate_de-en_validation.csv'
    }
    max_seq_len = 50
    batch_size = 64
    encoding_scheme = 'cl100k_base'