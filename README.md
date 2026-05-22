# Επεξεργασία Σημάτων Φωνής και Ήχου 2025-26

A simple voice activity detection (VAD) system written in Python.

## Modules

The system is quite modular, some would say "overly modular" 👀👀. I tried to separate each clear concept into its own Python module. Because this is a university project, I haven't taken the time to refactor the project to use Python packages for even more separation and organization, but I don't consider it necessary due to the project's size.

Overall, the system contains *two* types of modules: **core** and **utility** modules. Their roles are self-explanatory.

### Core Modules

**loader.py**
is responsible for loading WAV files into memory and ensuring they have a consistent data representation and sample rate. 

**preprocessor.py**
takes raw audio data and does some clean-up to normalize the data, for example, it boosts high frequencies (to learn why, see the documentation in the module). **Mainly**, it is responsible for **windowing**!

**extractor.py**
deals with per-frame feature extraction using Fourier Transforms, Mel-Frequency Cepstrum Coefficients (MFCCs), Zero Crossing Rate (ZCR) and more!

**dataset.py**
builds the dataset that will be used for training the classifiers downstream. This entails processing the "training" speech/noise files, mixing them together at different ratios, and then for each one extracting their features (using `extractor.py`). The result is a feature matrix `X_train` and the corresponding labels `y_train`.

**classifier.py**
contains the classifiers that are trained by the data from `dataset.py` and used to predict test data. As per the project's assignment, two classifiers are implemented: **k-NN** (k-nearest neighbours) and **MLP** (Multi-Layer Perceptron).

**postprocessor.py**
is the final crucial step of the pipeline. It takes in the "harsh" predictions from a classifier and refines them. This mainly consists of eliminating very short noise segments and adding some extra duration to certain speech segments. You can easily skip this step to see the "harshness" if no post-processing is applied on the predictions.

**evaluator.py**
is responsible for showing statistics and scoring the performance of the classification.

**files.py**
defines the operations that read from/write to CSV files the predicted speech/noise segments (called foreground/background respectively). 

### Utility Modules

**cache.py**
is a abstraction over the ``pickle`` package, used for storing trained models on disk for re-use (if possible).

**configuration.py** handles everything configuration (duh). Environment variables and CLI arguments are currently supported for configured the program. This module also sets the logging configuration.

---

Some modules have not been discussed due to their simplicity or obviousness

## Setup

The setup is the ordinary Python setup. It is recommended to create virtual environmment to ensure proper versions of libraries are being used. If you have the libraries mentioned in `requirements.txt` (under `src`), then you can try running the program without a venv

The short tutorial assumes a Unix-based shell (bash, zsh, etc...). If you are using the Windows terminal or Powershell, search how to create a venv in Python. The resources are plenty!

To setup a virtual environment, follow these steps:

**1 - Create virtual environment directory and contents**

```sh
src> python -m venv .venv
```

If `python` is not a command, it could possibly be because it is not aliased. In any case, if that happens, try:

```sh
src> python3 -m venv .venv
```

**2 - Activate/Source the virtual environment**

After you've created the environment, source it:

```sh
src> source .venv/bin/activate
```

**3 - Install libraries**

```sh
src> pip install -r requirements.txt
```
