## PME4 Dataset Summary
The PME4 is a posed multimodal emotion dataset with four modalities (PME4): audio, video, EEG, and EMG. 
Data were collected from 11 human subjects (five female and six male individuals) who were students in acting after informed consent was obtained. 
This dataset consists of recognizing the six basic human emotions (anger, fear, disgust, sadness, happiness, and surprise) plus a neutral emotion for a total of seven emotions.

PME4 can be accessed and downloaded for research purposes at https://doi.org/10.6084/m9.figshare.18737924

### Dataset File Listing
1. PME4_dataset_configs.csv
    - Each row is detail information for each trial
    - Columns
      - `subject` - subject number from 1 to 11
      - `trial` - trial number of each subject from 1 to 359 
      - `emotion_num` - emotion number of the corresponding trial
      - `emotion` - emotion string of the corresponding trial
      - `speech_start_time` - start time of the speech in second
      - `speech_stop_time` - stop time of the speech in second 
      - `speech_duration` - total time of the speech in second 
      - `audio_wav_filepath` - filepath of the wav data of corresponding trial
      - `audio_mfcc_100ms_50ms_filepath` - filepath of the mfcc features of the audio data with 100ms window and 50ms overlap for corresponding trial
      - `audio_mfcc_20ms_10ms_filepath` - filepath of the mfcc features of the audio data with 20ms window and 10ms overlap for corresponding trial 
      - `raw_eeg_filepath` - filepath of the eeg data of corresponding trial
      - `raw_emg_filepath` - filepath of the emg data of corresponding trial
      - `processed_eeg_filepath` - filepath of the filtered eeg data of corresponding trial
      - `processed_emg_filepath` - filepath of the mfcc features of the emg data with 2000ms window and 1000ms overlap for corresponding trial
      - `face_vgg_16_features_filepath` - filepath of the vgg16 features of the image sequences data for corresponding trial
        - some cells might be None, since not all subjects give us the consent to publish their video information.
2. Subject Folder
    - zip file from s01.zip to s11.zip, all with same structure
    - subject folder structure, same for s01 to s11
    ```
    s01
    └───t001
    │   │   s01_t001.wav (882KB) - raw audio wav file
    │   │   s01_t001_audio_mfcc_window20ms_overlap10ms.npy (48 KB)  - MFCC features of pre-processed audio data
    │   │   s01_t001_audio_mfcc_window100ms_overlap50ms.npy (10 KB) - MFCC features of pre-processed audio data
    │   │   s01_t001_raw_eeg_5kHz.npy (1.6 MB) - raw EEG file
    │   │   s01_t001_raw_emg_5kHz.npy (1.2 MB) - raw EMG file
    │   │   s01_t001_processed_eeg_1kHz.npy (320 KB) - Preprocessed EEG file
    │   │   s01_t001_processed_emg_5kHz_mfcc_window2000ms_overlap1000ms.npy (4 KB) - MFCC features of filtered EMG data
    │   └── s01_t001_face_vgg16_features.npy (3 MB) - VGG16 features of pre-processed image sequences (only existed if subjects give consent)
    └───t002
        │   ...
    ```

### Dataset File Details
#### Raw Audio file
These are original audio recordings in wav format, with 44.1kHz sampling rate for 5 seconds. 
The subject speak the generic sentence “The sky is green”. 

#### MFCC features of pre-processed audio data
1. Audio segmentation to extract 3-second speech duration interval - evenly expanded 1.5 seconds from center location of the speech
   - Speech duration ranging from 0.75 seconds to 3 seconds.
2. Normalized the audio segment.
3. Applied hamming window to remove the noise.
4. Extract 20 most significant mel-frequency cepstral coefficients

- Two different Hamming window sizes
  1. 20ms intervals with 10ms offsets 
  2. 100ms intervals with 50ms offsets

#### Raw EEG file
These are original EEG recordings in npy files, each with 8 recorded channels at 5kHz. 
The table below gives the EEG channel names for corresponding channel number.

<table>
  <tr>
    <th> Channel No. </th>
    <td align="center">1</td>
    <td align="center">2</td>
    <td align="center">3</td>
    <td align="center">4</td>
    <td align="center">5</td>
    <td align="center">6</td>
    <td align="center">7</td>
    <td align="center">8</td>
  </tr>
  <tr>
    <th>Channel Name</th>
    <td>F3</td>
    <td>Fz</td>
    <td>F4</td>
    <td>Cz</td>
    <td>P3</td>
    <td>Pz</td>
    <td>P4</td>
    <td>O2</td>
  </tr>
</table>


#### Preprocessed EEG file
1. The raw EEG data was downsampled to 1KHz.
2. EMG artifacts were removed.
3. A bandpass frequency filter from 0.1 - 30.0Hz was applied.

#### Raw EMG file
These are original EMG recordings in npy files, each with 6 channels at 5kHz. 
The table below gives the EMG channel names for corresponding channel number.

| Channel No. |Channel Name|
|:----------:|:-------------:|
| 1 | depressor anguli oris |
| 2 | zygomaticus major |
| 3 | levator labii superioris alaeque nasi |
| 4 | levator labii superioris |
| 5 | procerus |
| 6 | occipitofrontalis |

#### Preprocessed EMG file
1. A bandpass frequency filter from 20.0 - 500.0Hz was applied.
2. MFCC extraction with window size 2000ms and 1000ms overlapping window.

#### VGG16 features of pre-processed image sequences
We cannot share the original image/video data due to privacy consideration. 
The original video frames with a resolution of 960 x 720 pixels at 10 FPS.

The VGG16 features is obtain through the following steps for each trail
1. Extracted an image sequence of 16 screenshots per trial, evenly sampled from the central 1.5 seconds during-speech window at 10FPS.
2. Cropped the face for each image and resized to 224x224 pixels
3. Pass into the pretrained VGG16 model to obtain the feature vector of 2048 elements per image

