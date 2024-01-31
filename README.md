# VLG-Project

GENDER RECOGNITION USING VOICE

THEORY:
Voice features extraction
The Mel-Frequency Cepstrum Coefficients (MFCC) are used here, since they deliver the best results in speaker verification. MFCCs are commonly derived as follows:

Take the Fourier transform of (a windowed excerpt of) a signal.
Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
Take the logs of the powers at each of the mel frequencies.
Take the discrete cosine transform of the list of mel log powers, as if it were a signal
The MFCCs are the amplitudes of the resulting spectrum.


Gaussian Mixture Model
According to D. Reynolds in Gaussian_Mixture_Models: A Gaussian Mixture Model (GMM) is a parametric probability density function represented as a weighted sum of Gaussian component densities. GMMs are commonly used as a parametric model of the probability distribution of continuous measurements or features in a biometric system, such as vocal-tract related spectral features in a speaker recognition system. GMM parameters are estimated from training data using the iterative Expectation-Maximization (EM) algorithm or Maximum A Posteriori(MAP) estimation from a well-trained prior model.

![genderspeaker (2)](https://github.com/khushangi/VLG-Project/assets/118016692/65e61739-a49e-434b-9bf4-50052bed2ab9)

Import necessary libraries (os and urllib.request).
Specify the full path for the directory where the dataset will be saved (directory_path).
Create the specified directory if it does not exist.
Specify the destination file path, including the file name (destination_file).
Define the URL of the SLR45 dataset (zip_url).
Download the dataset from the specified URL and save it to the destination file using urllib.request.urlretrieve.
Print a success message indicating that the SLR45 dataset has been downloaded successfully.


1.Script defines a FileManager class to extract files from a tar.gz archive. It initializes with the path to the compressed file, extracts contents to a specified folder, and prints a success message. In the main block, an instance is created for a specific dataset, and files are extracted to a designated folder. Exceptions during extraction are caught and displayed as error messages.

2.Script defines a FileManager class that facilitates file operations. It extracts files from a tar.gz archive, creates folders, and moves files between them. In the main block, it's utilized to extract data from a specific dataset path, create training and testing folders, and organize files accordingly. Any exceptions during these operations are caught and displayed as error messages.


3.Script defines a FileManager class to manage file operations. It extracts files from a tar.gz archive, creates folders, moves files between them, and tracks moved files. In the main block, it is used to extract data from a specific dataset path, create training and testing folders, organize files, and print the names of files moved to the training folder. Any exceptions during these operations are caught and displayed as error messages.


4.Script defines a FileManager class to manage file operations and a function organize_dataset to organize files based on gender information in their filenames. In the main block, it extracts a compressed dataset, creates folders, organizes files by gender, and prepares for further processing (e.g., Gaussian Mixture Model training). The script utilizes the os, tarfile, shutil, and sklearn.mixture libraries for file and dataset management. Any exceptions during these operations are caught and displayed as error messages.


5.Script defines a `GenderGMMTrainer` class that uses Gaussian Mixture Models (GMM) to train gender-specific models for audio data. It utilizes the Librosa library for audio processing. In the main block, it initializes the trainer, trains GMM models for each gender using audio features extracted with Librosa, and optionally saves the trained models. Key points are as follows:

1. GenderGMMTrainer Class:
    - Initializes with the number of components for GMM and an empty dictionary to store trained models.
    - `train_models(self, dataset_path)`: Trains GMM models for each gender using audio features extracted from the provided dataset.
    - `extract_features(self, audio_path)`: Extracts MFCC (Mel-frequency cepstral coefficients) features from an audio file using Librosa.

2. Main Block:
    - Specifies the path to the organized dataset (`dataset_path`).
    - Creates an instance of `GenderGMMTrainer`.
    - Calls `train_models` to train GMM models for each gender using the provided dataset.
    - Optionally, saves the trained models.

script assumes that the dataset is organized with subfolders for each gender. Any exceptions during feature extraction are caught and displayed as error messages.

6.Script defines a `FeaturesExtractor` class for extracting MFCC (Mel-frequency cepstral coefficients) features from audio files using the Librosa library. It also includes a function `train_gmm_models` that utilizes Gaussian Mixture Models (GMM) from scikit-learn to train gender-specific models based on the extracted features. Key points are as follows:

1.FeaturesExtractor Class:
    - Initializes with parameters for MFCC extraction (number of coefficients, FFT size, hop length).
    - `extract_features(self, audio_path)`: Extracts MFCC features from an audio file using Librosa, handling potential exceptions.

2.train_gmm_models Function:
    - Utilizes the `FeaturesExtractor` to extract MFCC features from audio files in an organized dataset.
    - Trains GMM models for each gender using scikit-learn's `GaussianMixture`.
    - Saves the trained models to a specified folder using `joblib`.
   
3.Main Block:
    - Specifies the path to the organized dataset (`dataset_path`), the folder to save GMM models (`model_folder`), and the number of GMM components (`num_components`).
    - Creates an instance of `FeaturesExtractor`.
    - Calls `train_gmm_models` to extract features and train GMM models based on the provided dataset.

The script assumes that the organized dataset contains subfolders for each gender. It saves trained GMM models to a specified folder and prints the paths.

7.Script defines a `FeaturesExtractor` class for extracting MFCC (Mel-frequency cepstral coefficients) features from audio files using the Librosa library. Key points are as follows:

- FeaturesExtractor Class:
  - Initializes with parameters for MFCC extraction (number of coefficients, FFT size, hop length).
  - `extract_features(self, audio_path)`: Extracts MFCC features from an audio file using Librosa, handling potential exceptions.
    - Loads the audio file using the `audioread` backend.
    - Extracts MFCCs using Librosa with specified parameters.
    - Transposes the matrix to have time along the rows and features along the columns.
    - Returns the extracted MFCC features.
    - Prints an error message if any exceptions occur during the feature extraction process.

This class is designed for extracting audio features, specifically MFCCs, and provides flexibility in configuring the parameters of the extraction process. The use of warnings and exception handling ensures robustness when loading audio files.

8.Script performs gender classification using pre-trained Gaussian Mixture Models (GMMs) on MFCC features extracted from audio files. Key points are as follows:

-Feature Extraction Function:
  - `extract_features(audio_path, num_mfcc, n_fft, hop_length)`: Loads an audio file using the `audioread` backend, extracts MFCCs using Librosa, and transposes the matrix. Returns the extracted MFCC features.

-GMM Models Loading Function:
  - `load_gmm_models(model_folder)`: Loads pre-trained GMM models for both genders from a specified folder using `joblib`. Returns a dictionary of loaded models.

-System Testing Function:
  - `test_system(models, audio_path)`: Extracts features from an audio file and tests the system by computing log-likelihood scores using the loaded GMM models. Predicts the gender based on the highest score.

-Main Block:
  - Specifies the paths to the testing dataset (`dataset_path`) and the folder containing GMM models (`model_folder`).
  - Loads GMM models using `load_gmm_models`.
  - Tests the system on each audio file in the testing dataset using `test_system`.
  - Prints the predicted gender for each tested audio file.

The script assumes that the testing dataset contains audio files for gender classification. It handles feature extraction errors and provides predicted gender labels for each tested audio file based on the pre-trained GMM models.

