# elec5305-project-530773753
## Environmental Sound Classification Using MFCC, Mel Spectrograms, and CNNs

## Project Overview

This project develops an environmental sound classification system using audio signal processing and machine learning techniques.

The system will classify urban audio recordings into ten environmental sound categories. Two different approaches will be implemented and compared:

1. An MFCC-based baseline model using a Multilayer Perceptron (MLP).
2. An improved model using log-Mel spectrograms and a two-dimensional Convolutional Neural Network (2D CNN).

The project will also investigate whether audio data augmentation can improve classification accuracy and robustness under noisy conditions.

## Dataset

This project uses the UrbanSound8K dataset.

UrbanSound8K contains 8,732 labelled audio recordings of up to four seconds. The recordings are divided into ten classes:

* Air conditioner
* Car horn
* Children playing
* Dog bark
* Drilling
* Engine idling
* Gun shot
* Jackhammer
* Siren
* Street music

Dataset website:

https://urbansounddataset.weebly.com/urbansound8k.html

The complete dataset is not included in this repository. Users must download it separately and place it in the local `data/` directory.

## Project Objectives

The main objectives of this project are to:

* Extract MFCC features from environmental sound recordings.
* Generate log-Mel spectrograms from audio recordings.
* Develop an MFCC-MLP baseline classifier.
* Develop a 2D CNN classifier using log-Mel spectrograms.
* Apply audio data augmentation techniques.
* Compare the accuracy and robustness of the two systems.
* Analyse classification errors using confusion matrices.

## Proposed Methodology

### Audio Preprocessing

The audio recordings will be:

* Converted to mono.
* Resampled to 22.05 kHz.
* Normalised by amplitude.
* Padded or cropped to four seconds.
* Divided according to the official UrbanSound8K folds.

### Baseline System

The baseline system will:

1. Extract approximately 40 MFCCs from each audio recording.
2. Calculate summary statistics such as the mean and standard deviation.
3. Combine the statistics into a fixed-length feature vector.
4. Train an MLP to classify the feature vectors into ten classes.

### Improved System

The improved system will:

1. Convert each recording into a 128-band log-Mel spectrogram.
2. Use the spectrogram as the input to a 2D CNN.
3. Apply convolution, batch normalisation, ReLU activation, max pooling, and dropout.
4. Produce a ten-class prediction using a softmax output layer.

### Data Augmentation

The following augmentation methods may be evaluated:

* Time shifting
* Random gain adjustment
* Background-noise mixing
* Pitch shifting
* Time stretching

## Evaluation Metrics

The models will be evaluated using:

* Classification accuracy
* Macro-averaged precision
* Macro-averaged recall
* Macro F1-score
* Per-class recall
* Confusion matrix
* Training time
* Average inference time
* Model size

The robustness of the models will also be evaluated under additional noise at 20 dB, 10 dB, and 0 dB SNR.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── train_mfcc_mlp.py
│   ├── train_mel_cnn.py
│   └── evaluate.py
├── results/
│   └── figures/
└── docs/
```

## Installation

Clone this repository:

```bash
git clone https://github.sydney.edu.au/YOUR_USERNAME/elec5305-environmental-sound-classification.git
cd elec5305-environmental-sound-classification
```

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The implementation is currently under development.

Planned commands:

```bash
python src/preprocess.py
python src/train_mfcc_mlp.py
python src/train_mel_cnn.py
python src/evaluate.py
```

Detailed instructions will be added after the implementation is completed.

## Expected Results

The project is expected to show that:

* MFCC features provide a simple and computationally efficient baseline.
* Log-Mel spectrograms preserve more time-frequency information.
* A 2D CNN can learn discriminative patterns from Mel spectrograms.
* Data augmentation can improve generalisation and noise robustness.
* Some acoustically similar classes, such as drilling and jackhammer, may remain difficult to distinguish.

## Current Progress

* [x] Project topic selected
* [x] Project proposal completed
* [ ] UrbanSound8K dataset prepared
* [ ] Audio preprocessing implemented
* [ ] MFCC-MLP baseline implemented
* [ ] Mel-spectrogram CNN implemented
* [ ] Data augmentation implemented
* [ ] Experimental evaluation completed
* [ ] Final report completed
* [ ] Video demonstration completed

## References

1. J. Salamon, C. Jacoby, and J. P. Bello, “A dataset and taxonomy for urban sound research,” in *Proceedings of the 22nd ACM International Conference on Multimedia*, 2014, pp. 1041–1044.

2. K. J. Piczak, “Environmental sound classification with convolutional neural networks,” in *Proceedings of the IEEE 25th International Workshop on Machine Learning for Signal Processing*, 2015, pp. 1–6.

3. J. Salamon and J. P. Bello, “Deep convolutional neural networks and data augmentation for environmental sound classification,” *IEEE Signal Processing Letters*, vol. 24, no. 3, pp. 279–283, 2017.

4. B. McFee et al., “librosa: Audio and music signal analysis in Python,” in *Proceedings of the 14th Python in Science Conference*, 2015, pp. 18–25.

## Author

**Name:** [Your full name]
**Student ID:** [Your SID]
**Course:** ELEC5305
**GitHub username:** [Your GitHub username]

## Project Status

This project is currently under development as part of the ELEC5305 course project.
