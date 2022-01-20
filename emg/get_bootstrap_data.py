from utils.bootstrap import bootstrap_data


for random_state in [1626123187,1626123197, 1626123206,1626123216, 1626123226]:
    bootstrap_data(data_type="processed_eeg", num_average=20, num_train_bootstrap_samples=400,
                   num_test_bootstrap_samples = 100, random_state=random_state, save_bootstrap_data= True,
                   one_hot_encode=True)

