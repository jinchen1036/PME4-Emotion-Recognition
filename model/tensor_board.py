import os
import tensorflow as tf
from keras.callbacks import TensorBoard

class TensorBoardWriter(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TensorBoardWriter, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TensorBoardWriter, self).set_model(model)

    # def set_model(self, model):
    #     # Setup writer for validation metrics
    #     self.val_writer = tf.summary.FileWriter(self.val_log_dir)
    #     super(TensorBoardWriter, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}

        with self.val_writer.as_default():
            for name, value in val_logs.items():
                tf.summary.scalar(name, value, step=epoch)
                self.val_writer.flush()

        # for name, value in val_logs.items():
        #     summary = tf.compat.v1.Summary()
        #     summary_value = summary.value.add()
        #     summary_value.simple_value = value
        #     summary_value.tag = name
        #     self.val_writer.add_summary(summary, epoch)
        # self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TensorBoardWriter, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TensorBoardWriter, self).on_train_end(logs)
        self.val_writer.close()