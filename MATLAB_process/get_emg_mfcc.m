filename = 'filter_emg_5kHz';
bootstrap_data_path = '';
% emg_mfcc_data = emg_single_mfcc(data_path,filename, 2000, 1000, 5000);

for random_state = ["1626123187"]
    filename = sprintf('bootstrap_filter_random%s_400sample_per_emotion_20samples_filter_emg_5kHz_test',random_state)
    emg_mfcc_data = emg_bootstrap_mfcc(bootstrap_data_path, filename, 2000,1000, 5000);
    
    
    filename = sprintf('bootstrap_filter_random%s_400sample_per_emotion_20samples_filter_emg_5kHz_train',random_state)
    emg_mfcc_data = emg_bootstrap_mfcc(bootstrap_data_path,filename, 2000,1000, 5000);
    
end


