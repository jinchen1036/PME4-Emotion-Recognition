import os

from utils.file_accessor import check_dirpath
from model.tensor_board import TensorBoardWriter


def fit_general_model(model, model_dir, model_name, train_data, test_data, train_labels, test_labels,
                      n_epoch=5, epoch=1000, bs=128, init_epoch=0):
    print("TRAIN FOR MODEL - %s/%s" % (model_dir, model_name))
    check_dirpath(os.path.join(model_dir, model_name))

    for i in range(n_epoch):
        model.fit(train_data, test_data,
                  initial_epoch=init_epoch + i * epoch,
                  epochs=init_epoch + (i + 1) * epoch,
                  batch_size=bs,
                  validation_data=(train_labels, test_labels),
                  shuffle=True,
                  verbose=2,
                  callbacks=[TensorBoardWriter(log_dir=os.path.join(model_dir, model_name), write_graph=False)])

        model.save(os.path.join(model_dir, f"{model_name}_ep{(init_epoch + (i + 1) * epoch)}.h5"))


def fit_model_w_train_only(model, model_dir, model_name, train_data, test_data, log_dir,
                           n_epoch=5, epoch=1000, bs=128, init_epoch=0):
    main_path = os.path.join(model_dir, log_dir)
    model_path = os.path.join(main_path, main_path)
    print("TRAIN FOR MODEL - %s" % model_path)
    modelName = "%s/%s" % (model_name, model_name)

    check_dirpath(model_path)

    for i in range(n_epoch):
        model.fit(train_data, test_data,
                  initial_epoch=init_epoch + i * epoch,
                  epochs=init_epoch + (i + 1) * epoch,
                  batch_size=bs,
                  shuffle=True,
                  verbose=2,
                  callbacks=[TensorBoardWriter(log_dir=model_path, write_graph=False)])
        model.save(os.path.join(model_path, f"{model_name}_ep{(init_epoch + (i + 1) * epoch)}.h5"))
