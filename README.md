# elec5305-project-530773753
## Environmental Sound Classification Using MFCC, Mel Spectrograms, and CNNs

## Project Overview

This project develops an environmental sound classification system using audio signal processing and machine learning techniques.

The system classifies urban recordings into ten sound categories. The original proposal compared an MFCC-MLP baseline with a log-Mel CNN. Following supervisor feedback, a temporal MFCC CNN was added to compare MFCC and log-Mel representations using the same CNN architecture.

The current implementation evaluates three models on the official UrbanSound8K ten folds:

1. Model A: mean/std aggregated MFCC features with an MLP.
2. Model B: time-resolved MFCC features with a CNN.
3. Model C: log-Mel spectrograms with the same CNN architecture.

Audio augmentation and noise robustness were proposed originally and remain future work.

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

The complete dataset is not included in this repository. Users must download it separately and extract it into `data/UrbanSound8K/`, containing `audio/fold1` through `audio/fold10` and `metadata/UrbanSound8K.csv`.

## Project Objectives

The main objectives of this project are to:

* Extract MFCC features from environmental sound recordings.
* Generate log-Mel spectrograms from audio recordings.
* Develop an MFCC-MLP baseline classifier.
* Develop a 2D CNN classifier using log-Mel spectrograms.
* Apply audio data augmentation techniques.
* Compare the accuracy and robustness of the two systems.
* Analyse classification errors using confusion matrices.

## Implemented Methodology

### Audio Preprocessing

The audio recordings are:

* Converted to mono.
* Resampled to 22.05 kHz.
* Processed without per-recording peak normalization; feature normalization uses training-fold statistics only.
* Padded or cropped to four seconds.
* Divided according to the official UrbanSound8K folds.

### Baseline System

The baseline system:

1. Extract 40 MFCCs from each audio recording.
2. Calculate summary statistics such as the mean and standard deviation.
3. Combine the statistics into a fixed-length feature vector.
4. Train an MLP to classify the feature vectors into ten classes.

### Improved System

The log-Mel CNN system:

1. Convert each recording into a 128-band log-Mel spectrogram.
2. Use the spectrogram as the input to a 2D CNN.
3. Apply convolution, batch normalisation, ReLU activation, max pooling, and dropout.
4. Produce a ten-class prediction from classifier logits; training uses cross-entropy loss.

### Temporal MFCC CNN

Model B uses 40 MFCC coefficients across time as input to the same CNN architecture as Model C. Comparing B and C examines representation effects. Comparing A and B changes both temporal aggregation and classifier architecture.

### Data Augmentation Planned

The following augmentation methods may be evaluated:

* Time shifting
* Random gain adjustment
* Background-noise mixing
* Pitch shifting
* Time stretching

## Evaluation Metrics and Scope

The original evaluation plan included:

* Classification accuracy
* Macro-averaged precision
* Macro-averaged recall
* Macro F1-score
* Per-class recall
* Confusion matrix
* Training time
* Average inference time
* Model size

Completed outputs include accuracy, Macro-F1, normalized confusion matrices and parameter counts. Per-model inference latency has not been measured. Noise testing at 20 dB, 10 dB and 0 dB SNR and augmentation remain planned; reproduction of a published baseline has not been confirmed.




## Original Expected Results

The original proposal anticipated the following; these expectations are not all established by the completed experiments:

* MFCC features provide a simple and computationally efficient baseline.
* Log-Mel spectrograms preserve more time-frequency information.
* A 2D CNN can learn discriminative patterns from Mel spectrograms.
* Data augmentation can improve generalisation and noise robustness.
* Some acoustically similar classes, such as drilling and jackhammer, may remain difficult to distinguish.

## Current Progress

* [x] Project topic selected
* [x] Project proposal completed
* [x] UrbanSound8K dataset prepared
* [x] Audio preprocessing implemented
* [x] MFCC-MLP baseline implemented
* [x] Mel-spectrogram CNN implemented
* [ ] Data augmentation implemented
* [x] Official ten-fold evaluation of all three models completed
* [x] Temporal MFCC CNN implemented
* [x] Representative real-audio analyses completed
* [x] Feedback TWO report prepared
* [ ] Final project report completed
* [ ] Video demonstration completed

## Current Experimental Results

| Model | Input | Parameters | Accuracy mean | Accuracy SD | Macro-F1 mean | Macro-F1 SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A MFCC-MLP | 80-dimensional mean/std MFCC summary | 19,274 | 60.2330% | 5.2844 pp | 0.606483 | 0.049931 |
| B MFCC-CNN | Temporal MFCC | 24,170 | 63.0037% | 7.4005 pp | 0.635801 | 0.066192 |
| C Log-Mel CNN | Log-Mel spectrogram | 24,170 | 64.2163% | 6.4850 pp | 0.652673 | 0.058329 |

SD is the standard deviation across the ten folds; pp means percentage points. No statistical significance has been established. Model C has the highest observed mean scores, but does not improve every class. Exact aggregate values are retained in [summary.json](results/cross_validation/summary.json).

Shared settings in [config.json](config.json): 22,050 Hz mono audio, four-second padding/cropping, FFT 1024, hop 512, 40 MFCC coefficients, 128 Mel bands, batch size 32, Adam learning rate 0.001, maximum 50 epochs and early-stopping patience 8. Each run uses one test fold, the next official fold for validation, and the other eight folds for training.

## Setup and Reproduction

From the repository root, create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and extract UrbanSound8K separately. Then inspect the dataset and build the features:

```powershell
python src/inspect_dataset.py
python src/dataset.py --build
python src/cnn_dataset.py --build both
```

Review the evaluation plan, run the official-fold experiments, and summarize saved results:

```powershell
python src/evaluate.py --dry-run
python src/evaluate.py --run
python src/evaluate.py --summarize
```

These commands reflect the script interfaces in this repository; they have not been rerun during the README restoration. Synthetic self-tests are software checks and must not be interpreted as UrbanSound8K performance.

## Repository Contents

* `src/`: feature extraction, dataset preparation, training, evaluation and sound analysis.
* `config.json`: shared experiment configuration.
* `requirements.txt`: environment dependencies.
* `results/cross_validation/`: aggregate results and comparison figures.
* `results/mlp/` and `results/cnn/`: model evaluation artifacts.
* `results/confusion_analysis/`: confusion analysis outputs.
* `results/audio_cases/`: drilling, jackhammer, engine-idling and air-conditioner analyses.
* [Feedback TWO PDF](ELEC5305_Feedback_TWO_Xiao_Hu.pdf) and [editable DOCX](ELEC5305_Feedback_TWO_Xiao_Hu.docx): the report with original visual evidence.

The dataset, virtual environment and generated feature caches are excluded from version control. Individual sound examples are illustrative rather than representative of entire classes. In particular, the air-conditioner example contains approximately 2.05 seconds of active audio padded to four seconds; its modulation plot includes padding and finite-window effects.

## References

1. J. Salamon, C. Jacoby, and J. P. Bello, "A dataset and taxonomy for urban sound research," ACM Multimedia, 2014.
2. K. J. Piczak, "Environmental sound classification with convolutional neural networks," IEEE MLSP, 2015.
3. J. Salamon and J. P. Bello, "Deep convolutional neural networks and data augmentation for environmental sound classification," IEEE Signal Processing Letters, 2017.
4. B. McFee et al., "librosa: Audio and music signal analysis in Python," Python in Science Conference, 2015.

## Author

**Name:** Xiao Hu
**Student ID:** 530773753
**Course:** ELEC5305
**GitHub username:** xihu0648

## Project Status

This project is currently under development as part of the ELEC5305 course project.
