from abc import ABC, abstractmethod

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras.models import load_model, save_model

from src.basemodel.metrics import metrics_dict
from src.util import get_models_dir


class SaveModelOnInterval(Callback):
    """
    Save the model with a specified frequency
    """

    def __init__(self, savefile, interval=10):
        super().__init__()
        self.interval = interval
        self.savefile = savefile

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and (epoch + 1) % self.interval == 0:
            print(f"[{self.__class__.__name__}] saving model")
            save_model(self.model, f"{self.savefile}_{epoch + 1}.h5", include_optimizer=True)


class CsvLearningRateAndTrainingTimePerEpoch(Callback):
    """
    Save on file learning rate and time spent per epoch
    """

    def __init__(self, filename):
        super(CsvLearningRateAndTrainingTimePerEpoch, self).__init__()

        self.filename = filename

        self._open_args = {'newline': '\n'}

        self.fieldnames = ["epoch", "lr", "train time"]
        self.csv_file = None
        self.csv_writer = None

        self.start = None

    def on_train_begin(self, logs=None):
        import os, io, csv
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("model need lr")
        if os.path.exists(self.filename):
            mode = "a"
        else:
            mode = "w"

        self.csv_file = io.open(self.filename, mode, **self._open_args)
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)

        with open(self.filename, 'r') as f:
            write_header = not bool(len(f.readlines()))
            if write_header: self.csv_writer.writeheader()

    def on_epoch_begin(self, epoch, logs=None):
        from datetime import datetime
        self.start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        from datetime import datetime
        import keras.backend as K

        end = datetime.now()
        diff = end - self.start

        lr = float(K.get_value(self.model.optimizer.lr))

        row = {"epoch": epoch, "lr": lr, "train time": str(diff)}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.csv_writer = None

    def __del__(self):
        if self.csv_file is None:
            return

        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()


class BaseModel(ABC):
    """
    Base class for define a model
    """

    def __init__(self,
                 name: str,
                 batch_size: int,
                 epochs: int,
                 early_stopping_patience=5,
                 save_on="val_loss",
                 save_on_interval=10,
                 last_epoch=0):
        """
        :param name: name of the model
        :param batch_size: size of the batch
        :param epochs: maximum number of epochs
        :param early_stopping_patience: patience for stopping after no improvement of @save_on
        :param save_on: what look for the improvement. Default: val_loss
        :param save_on_interval: frequency for saving a model independently from @save_on
        :param last_epoch: last ended training epoch. Default: 0
        """
        self.batch_size = batch_size
        self.epochs = epochs

        self.name = name
        self.model_dir = get_models_dir() / self.name
        self.model_name = f"{name}_best" if last_epoch == 0 else f"{name}_{last_epoch}"
        self.model_file = self.model_dir / f"{self.model_name}.h5"

        self.last_epoch = last_epoch
        self.early_stopping_patience = early_stopping_patience
        self.save_on = save_on
        self.save_on_interval = save_on_interval

        self._model = None

        if not self.model_dir.exists():
            self.model_dir.mkdir()

        if self.model_dir.exists() and self.model_file.exists():
            self.set_custom_objects()
            self._model = load_model(self.model_file, custom_objects=self.custom_objects, compile=True)
            print("Previous model loaded.")

        self.callbacks = self.set_callbacks()

    def set_callbacks(self):
        """
        Define the checkpoints to use
        :return: list of checkpoints
        """
        model_checkpoint = ModelCheckpoint(str(self.model_dir / f"{self.name}_best.h5"), monitor=self.save_on,
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False, mode="auto")
        early_stopping = EarlyStopping(monitor=self.save_on, patience=self.early_stopping_patience, verbose=1,
                                       mode="auto")
        csv_logger = CSVLogger(str(self.model_dir / f"{self.name}.log.csv"), append=True)
        save_on_interval = SaveModelOnInterval(self.model_dir / self.name, self.save_on_interval)
        csv_lr = CsvLearningRateAndTrainingTimePerEpoch(str(self.model_dir / f"{self.name}.lr.log.csv"))
        return [model_checkpoint, early_stopping, csv_logger, save_on_interval, csv_lr]

    def set_custom_objects(self):
        """
        Set the custom objects loaded by the model
        By default there are metrics
        It's possible to add losses and other elements

        USAGE:
            self.custom_objects.update({"key":"value"})

        NOTE:
            It's important to call super(...).set_custom_objects() otherwise the metrics are not loaded

        """
        # Add base custom object
        self.custom_objects = metrics_dict

    @abstractmethod
    def _set_model(self):
        """
        Set the architecture of the model in self.model
        :return:
        """
        raise NotImplementedError(f"_set_model not implemented in {self.__class__.__name__}")

    @abstractmethod
    def compile(self):
        """
        Set the optimizer, loss, metrics of the model
        :return:
        """
        raise NotImplementedError(f"compile not implemented in {self.__class__.__name__}")

    @abstractmethod
    def _data(self):
        """
        Method that handle how data are retrieved from the model.
        :return: (train_gen,val_gen,test_gen) all as Sequence
        """
        return (None, None, None)

    def train(self):
        """
        Train the model.
        """
        if self._model is None:
            self._set_model()

        self.compile()

        (train_gen, val_gen, _) = self._data()

        history = self._model.fit_generator(generator=train_gen,
                                            validation_data=val_gen,
                                            epochs=self.epochs,
                                            callbacks=self.callbacks,
                                            initial_epoch=self.last_epoch)

        from basemodel.common import plot_history
        plot_history(self.model_dir, history)
        return history

    def evaluate(self):
        """
        Evaluate the model.
        """
        if self._model is None:
            raise ValueError("self._model is None. Error loading model or you have to train the network")

        (_, _, test_gen) = self._data()
        results = self._model.evaluate_generator(test_gen, self.batch_size)
        print(results)
        with open(self.model_dir / "evaluation", "w") as f:
            f.writelines(",".join([str(result) for result in results]))

    def test(self):
        """
        Test the model
        """
        from random import randint

        if self._model is None:
            raise ValueError("model is None.")

        (_, _, test_gen) = self._data()
        batch_index = randint(0, test_gen.__len__())
        batch_index = 12345 % test_gen.__len__()
        X_input, Y_true = test_gen.__getitem__(batch_index)

        Y_pred = self._model.predict(X_input, batch_size=self.batch_size)
        import matplotlib.pyplot as plt
        n = 3
        fig, axes = plt.subplots(n, 3, sharex='col')
        for i in range(n):
            axes[i][0].imshow(X_input[i])
            axes[i][0].set_title("Input")
            axes[i][1].imshow(Y_pred[i])
            axes[i][1].set_title("Predicted image")
            axes[i][2].imshow(Y_true[i])
            axes[i][2].set_title("True image")
        path = self.model_dir / "test.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Image generated at: {path}")

    def summary(self):
        """
        Print the summary of the model
        Plot the model on a file
        Save the model on a file without optimizer
        """
        if self._model is None: self._set_model()
        print(self._model.summary())
        from keras.utils import plot_model
        plot_model(self._model, self.model_dir / "model.png", show_shapes=True, rankdir="LR")
        if not self.model_file.exists(): self._model.save(self.model_file, overwrite=True, include_optimizer=False)

    def __str__(self):
        return self.name
