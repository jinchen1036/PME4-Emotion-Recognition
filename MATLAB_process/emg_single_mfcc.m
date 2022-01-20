function [emg_mfcc_data] = emg_single_mfcc(data_path, data_name, Tw,Ts, fs)
    addpath('') % datapath

%     fs = 1000;
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 20;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 20;               % lower frequency limit (Hz)
    HF = 500;              % upper frequency limit (Hz)

    emg_bootstrap_data = readNPY(strcat(data_name,'.npy'));
    
    for trial = 1:size(emg_bootstrap_data,1)
        for channel = 1:size(emg_bootstrap_data,2)
            data_seg = reshape(emg_bootstrap_data(trial,channel,:),1,[]);
            [norm_MFCCs, norm_FBEs, norm_frames ] = mfcc(data_seg,fs, Tw, Ts, alpha, @hamming, [LF HF], M, C, L );
            emg_mfcc_data(trial,channel,:,:) = norm_MFCCs.';
        end
    end
    filename = sprintf('%s/%s_mfcc_%dms_%dms.npy', data_path,data_name, Tw, Ts);
    writeNPY(emg_mfcc_data, filename);
end