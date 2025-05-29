# DanQ for Plant Genome
**DanQPlant** is a deep learning model that combines a convolutional neural network (CNN) with a bidirectional long short-term 
memory network (BLSTM). It was originally proposed by Quang and Xie in 2016 and is used to predict functional regions of DNA 
sequences, especially in gene regulation tasks such as transcription factor binding sites and epigenetic marks.

## Model Structure

<img width="189" alt="image" src="https://github.com/user-attachments/assets/7cfa9194-ddab-4457-b91c-a4a0f9f42abd" />


Figure 1. DanQ model.

## Key Features
- **Input layer**  
  The original DNA sequence is usually input in one-hot encoding form, for example, a sequence of 1,024 bp in length is
  converted into a 1024×4 matrix.

- **Convolutional layer（CNN）**  
  Extracting local sequence features is equivalent to simulating the recognition function of motifs in organisms. Using multiple
  convolution kernels, each kernel can learn different motif patterns.

- **MaxPooling layer**  
  Dimensionality reduction and retention of the strongest local features are usually used to compress convolution results and improve
  computational efficiency.

- **Bidirectional Long Short-Term Memory Network（BLSTM）**  
 Capture long-distance dependency information in the sequence. Compared with traditional CNN models (such as DeepSEA), after adding
 Bi-LSTM, the model can understand the contextual relationship between sequence fragments.

- **Fully connected layer (Dense) and output layer**  
  Finally, the probability of each label is output through the sigmoid activation function to support multi-label (binary) classification.

## Quick Start
### Install and use it by Git

```bash
git clone https://github.com/njuxka2333/DanQPlant.git
cd DanQPlant
conda create -n danq_env python=3.8
conda activate danq_env
pip install -r requirements.txt

```

### Step 1: Prepare FASTA Input and Generate DeepSEA dataset

- **Sequence extraction**: Fixed-length sequences (1024 bp) were extracted from ChIPHub data and divided into training/validation/test sets
  based on chromosome number.
- **one-hot encoding**: Convert the DNA sequence to a one-hot vector of shape `(L, 4)` (A, C, G, T correspond to `[1,0,0,0]`, etc.).
- **Label binary classification**: Convert the label information into a 1×n (n is the number of labels) vector, where 1 indicates the label exists and 0 indicates it does not exist.

Example command:
```bash
python build_DeepSEA_data.py\
--tag_flie original/tag.txt 
--train_valid_file original_data/mergedtag_1024_512.fa 
--test_file original_data/mergedtag_1024_500.fa 
-- train
--train_filename data/train.mat \ 
--valid_filename data/valid.mat \ 
--test_filename  data/test.mat \ 

```
- **tag_flie**: tag file path
- **train_valid_file original_data**: fasta data path for training and validation.
- **train_valid_file**: fasta data path for training and validation.
- **test_file**: fasta data for test.
- **train_file**: train.mat file path
- **valid_file**: valid.mat file path
- **test_file**: test.mat file path

---

### Step 2: Constructing the DanQ model structure
DanQ model
```python
forward_lstm = LSTM(units=256, return_sequences=True)
      backward_lstm = LSTM(units=256, return_sequences=True, go_backwards=True)
      brnn = Bidirectional(forward_lstm, backward_layer=backward_lstm)

      logging.info('building model')

      model = Sequential()
      model.add(Conv1D(filters=256,
                       kernel_size=26,
                       input_shape=(1024, 4),
                       padding="valid",
                       activation="relu",
                       strides=1,
                       groups=1))
      model.add(BatchNormalization())
      model.add(MaxPooling1D(pool_size=13, strides=13))
      model.add(Dropout(0.2))
      model.add(brnn)
      model.add(BatchNormalization())
      model.add(Dropout(0.5))
      model.add(Flatten())
      model.add(Dense(units=745,activation='relu',kernel_regularizer=l2(0.0005)))
      model.add(BatchNormalization())
      model.add(Dense(units=num_label,activation='sigmoid'))
```
- num_labels depends on the dataset (for example, DeepSEA is 919; for plants, the number of Histone labels can be customized).
- The optimizer commonly uses RMSprop and the loss function is binary crossentropy.

---

### Step 3: Model training

- Use model.fit(...) or a custom DataGenerator to implement batch training;
- Often used with callbacks such as EarlyStopping and ModelCheckpoint;
- When the data set is large, it is recommended to use tf.data.Dataset or a custom generator to support multi-threaded reading.

Example command:
```bash
python DanQ_train.py

```

---

### Step 4: Model Evaluation and Visualization

The test script will output the following:
- Generate ROC curve and PRAUC curve
- Calculate the average F1 score, average precision and average recall
- Calculate the Pearson correlation coefficient and Spearman correlation coefficient between y_pred and y_true

Example command:
```bash
python DanQ_test.py
```
