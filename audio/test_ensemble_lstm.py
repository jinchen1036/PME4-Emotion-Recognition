# must pre train the lstm models (check the audio_main.py to train the individual lstm models)

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score

from model.lstm_model import LSTM
from utils.settings import Settings
from utils.data_formater import generate_lstm_result
from utils.data_loader import load_train_test_cv_split_data

def test_ensemble_lstm_cv(data_type:str, settings: Settings,trained_epoch = 60, cv_n = 5, number_models: int = 20):
    all_model_result = []
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=True,
                                                                                          presaved_index=True):
        predictions = []
        (train_size, timesteps, num_features) = train_data.shape
        (_, num_classes) = train_labels.shape

        for model_num in range(1, number_models+1):
            model_name = LSTM.construct_model_name(lr=settings.learning_rate, lstm_units=settings.lstm_units,
                          l1_value=settings.l1, l2_value=settings.l2,dropout=settings.dropout)
            model_name = f"cv{cv_i}_{timesteps}timesteps_Model{'%02d' % model_num}_LSTM{model_name}"
            model_dir = os.path.join(os.getenv("PME4_DATA_PATH"), "model", data_type)
            model = load_model(os.path.join(model_dir,model_name,f"{model_name}_ep{trained_epoch}.h5"))

            prediction = model.predict(test_data)
            accuracy = accuracy_score(np.argmax(test_labels,axis=1) ,np.argmax(prediction,axis=1) )
            print("Processing the CV %d, model %d epoch %d -- Acc %.05f"%(cv_i, model_num,trained_epoch, accuracy))
            predictions.append(prediction)
            all_model_result.append(generate_lstm_result(data_type=data_type,cv_i=cv_i,model_num=model_num, trained_epoch=trained_epoch, accuracy=accuracy))
        predictions = np.array(predictions)
        mean_acc = accuracy_score(np.argmax(test_labels,axis=1) ,np.argmax(predictions.mean(axis=0),axis=1))
        all_model_result.append(generate_lstm_result(data_type=data_type,cv_i=cv_i,model_num="sum", trained_epoch=trained_epoch, accuracy=mean_acc))

        print("Mean Probability Accuracy %.5f"%mean_acc)

    model_result_pd = pd.DataFrame(all_model_result)
    model_result_pd.to_csv(os.getenv("PME4_DATA_PATH"),  f"{data_type}_ensemble_lstm_result.csv")