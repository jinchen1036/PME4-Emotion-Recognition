function [emg_mfcc_data] = emg_mfcc_concat(data, datapath, dataname, Tw,Ts)
    addpath('') % datapath

    fs = 5000;
%      Tw = 2000;                % analysis frame duration (ms)
%      Ts = 1000;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 20;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 20;               % lower frequency limit (Hz)
    HF = 500;              % upper frequency limit (Hz)
    
    fn = fieldnames(data);
    for sub =1:numel(fn)
        emg_filter_data = readNPY(strcat(fn{sub},'_epoch_emg_filter_5kHz.npy'));

        subject_data = data.(fn{sub});
        emo_fn = fieldnames(subject_data);
        for emo = 1:numel(emo_fn)
            fprintf('Subject %d, Emotion %d\n', sub, emo)
            train_trial_index = subject_data.(emo_fn{emo}).train_trials ;    % not add 1 because the index begin with 1
            test_trial_index = subject_data.(emo_fn{emo}).test_trials;

            train_emg_data = emg_filter_data(train_trial_index,:,:);
            test_emg_data = emg_filter_data(test_trial_index,:,:);

            train_group_index = subject_data.(emo_fn{emo}).train_index_groups + 1;    % add 1 because the index in python begin with 0
            test_group_index = subject_data.(emo_fn{emo}).test_index_groups + 1;

            for trial = 1 : size(train_group_index,1)
                group_data = train_emg_data(train_group_index(trial,:),:,:);
                for channel = 1 : size(group_data,2)
                    emg_bootstrap_concate_data = reshape(group_data(:,channel,:), size(group_data,1),[]);
                    emg_bootstrap_concate_data = reshape(emg_bootstrap_concate_data',1,[]);
                    [norm_MFCCs, norm_FBEs, norm_frames ] = mfcc(emg_bootstrap_concate_data,fs, Tw, Ts, alpha, @hamming, [LF HF], M, C, L );
                    emg_bootstrap_concate_mfcc_train_data(sub,emo,trial,channel,:,:) = norm_MFCCs.';
                end
            end

           for trial = 1 : size(test_group_index,1)
                group_data = test_emg_data(test_group_index(trial,:),:,:);
                for channel = 1 : size(group_data,2)
                    emg_bootstrap_concate_data = reshape(group_data(:,channel,:), size(group_data,1),[]);
                    emg_bootstrap_concate_data = reshape(emg_bootstrap_concate_data',1,[]);
                    [norm_MFCCs, norm_FBEs, norm_frames ] = mfcc(emg_bootstrap_concate_data,fs, Tw, Ts, alpha, @hamming, [LF HF], M, C, L );
                    emg_bootstrap_concate_mfcc_test_data(sub,emo,trial,channel,:,:) = norm_MFCCs.';
                end
           end
        end

        filename = sprintf('%s/%s_5kHz_concate_mfcc_%dms_%dms_train.npy', datapath,dataname, Tw, Ts);
        writeNPY(emg_bootstrap_concate_mfcc_train_data, filename);

        filename = sprintf('%s/%s_5kHz_concate_mfcc_%dms_%dms_test.npy', datapath,dataname, Tw, Ts);
        writeNPY(emg_bootstrap_concate_mfcc_test_data, filename);

    end
end