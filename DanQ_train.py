import os
import h5py
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DanQ_model import DataGenerator, create_model, LoggingCallback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'
np.random.seed(1337)  # for reproducibility

# Logging configuration
logging.basicConfig(level=logging.INFO)
logging.info('loading data')

trainmat = h5py.File('data/train.mat', 'r')
logging.info('train.mat loaded')
validmat = h5py.File('data/valid.mat', 'r')
logging.info('valid.mat loaded')

batch_size = 32 # batch size
epochs=5
lr = 8.674559776324777e-05

X_train, y_train = np.array(trainmat['trainxdata']), np.array(trainmat['traindata'])
valid_x, valid_y = np.array(validmat['validxdata']), np.array(validmat['validdata'])


logging.info('building model')
model = create_model(lr=lr)  # lr


# Data Generator
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)
val_generator = DataGenerator(valid_x, valid_y, batch_size=batch_size, shuffle=True, normalize=True, augment_rc=True)

# checkpointer
checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)

# earlystopper
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# logging callback
logging_callback = LoggingCallback()

# training model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,  # epoch number
    callbacks=[checkpointer, logging_callback],
    workers=1,
    use_multiprocessing=False
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
