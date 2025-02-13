from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,Callback
import logging
import pickle


# this override saves model as .json separately from weights as .h5 to work around cudnn failed to initialize error 
# from running model.predict off a model reloaded from previous model.save
class ModelCheckpoint_json(ModelCheckpoint):

    def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               **kwargs):
        ModelCheckpoint.__init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=save_best_only,
               save_weights_only=save_weights_only,
               mode='auto',
               save_freq='epoch',
               **kwargs)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                    int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s' % (epoch + 1, self.monitor,
                                                            self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                model_json = self.model.to_json()
                                with open(filepath+'.json','w') as fb:
                                    fb.write(model_json)
                                    fb.close()
                                self.model.save_weights(filepath+'.h5', overwrite=True)
                                with open(filepath+'-hist.pickle','wb') as fb:
                                    trainhistory = {"history": self.model.history.history,"params": self.model.history.params}
                                    pickle.dump(trainhistory,fb)
                                    fb.close()
                                # self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                    (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        model_json = self.model.to_json()
                        with open(filepath+'.json','w') as fb:
                            fb.write(model_json)
                            fb.close()
                        self.model.save_weights(filepath+'.h5', overwrite=True)
                        with open(filepath+'-hist.pickle','wb') as fb:
                            trainhistory = {"history": self.model.history.history,"params": self.model.history.params}
                            pickle.dump(trainhistory,fb)
                            fb.close()

                self._maybe_remove_file()
            except IOError as e:
            # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]):
                    raise IOError('Please specify a non-directory filepath for '
                                'ModelCheckpoint. Filepath used is an existing '
                                'directory: {}'.format(filepath))

# class callbacktest(Callback):
#     def __init__(self,model):
#         Callback.__init__(self)
#         self.model = model

#     def set_model(self, model):
#         self.model = model
#         if self._history:
#             model.history = self._history
#         for callback in self.callbacks:
#             callback.set_model(model)

#     def _save_model(self):
#         model_history = self.model.history
#         with open('model-history.pickle') as fb:
#             pickle.dump(model_history,fb)
#             fb.close()

