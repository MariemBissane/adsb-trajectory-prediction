# Aircraft Trajectory Prediction using ADS-B Data

## Project Overview

This project focuses on predicting aircraft trajectories using Machine Learning and Deep Learning techniques applied to real-world ADS-B aviation data from the OpenSky Network.
This is a team project carried out at AIAC by a team of six students, where I was in charge of the **Deep Learning part** (GRU, LSTM and TCN models).

The objective is to forecast future aircraft positions (latitude, longitude, altitude) from historical flight trajectories in order to support:

* Air Traffic Management (ATM)
* Flight anomaly detection
* Route optimization
* Aviation data analysis

The project implements a complete AI pipeline including:

* Data collection
* Exploratory Data Analysis (EDA)
* Data cleaning
* Feature engineering
* Machine Learning models
* Deep Learning architectures
* Hyperparameter optimization

---

## Dataset

Source: OpenSky Network Historical Dataset

* Date used: July 3rd, 2017
* Time range: 10:00 → 17:00 UTC
* Raw observations: ~17.8 million ADS-B records
* Final processed dataset: ~11.7 million clean records
* Continuous trajectories generated: ~40,205

ADS-B variables include:

* Latitude / Longitude
* Velocity
* Heading
* Vertical rate
* Altitude
* Aircraft identifier (icao24)
* Callsign
* Timestamp

---

## Data Processing Pipeline

### 1. Exploratory Data Analysis (EDA)

Performed analyses:

* Missing values analysis
* Correlation heatmaps
* Pair plots
* Velocity-altitude relationships
* Temporal traffic analysis

### 2. Data Cleaning

A 16-step cleaning pipeline was implemented:

* Removal of invalid coordinates
* Missing trajectory filtering
* Altitude fusion (`geoaltitude` + `baroaltitude`)
* Duplicate removal
* Chronological sorting
* Quality validation

### 3. Feature Engineering

Generated features include:

* Ground speed
* Acceleration
* Turn rate
* Vertical speed
* Distance traveled
* Rolling statistics
* Flight phase classification
* Velocity vector decomposition

### 4. Temporal Resampling

Trajectories were resampled every 10 seconds to create homogeneous sequences suitable for sequential deep learning models.

---

## Machine Learning Models

Implemented models:

| Model                       | Task                        |
| --------------------------- | --------------------------- |
| Decision Tree Regressor     | Trajectory prediction       |
| Random Forest Regressor     | Trajectory prediction       |
| Gradient Boosting Regressor | Trajectory prediction       |
| PCA + SVM (RBF)             | Flight phase classification |

### Best ML Result

Gradient Boosting achieved:

* Haversine Error: ~4.5 km
* Strong trajectory tracking performance
* Good robustness on long trajectories

---

## Deep Learning Models

Implemented architectures:

* GRU (Gated Recurrent Unit)
* LSTM (Long Short-Term Memory)
* TCN (Temporal Convolutional Network)

### Best DL Model

TCN achieved the best baseline performance:

* Average trajectory deviation: ~15.7 km
* Excellent long-sequence modeling
* Stable training on CPU

Hyperparameter tuning was performed using Optuna.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Optuna
* Matplotlib
* Seaborn
* Jupyter Notebook


---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run notebooks or scripts for:

1. Data preprocessing
2. Trajectory cleaning
3. Feature engineering
4. Model training
5. Evaluation and visualization

---

## Results

### Machine Learning

| Model             | Haversine Error |
| ----------------- | --------------- |
| Decision Tree     | 20.6 km         |
| Random Forest     | 16.8 km         |
| Gradient Boosting | 4.5 km          |

### Deep Learning

| Model | Average Deviation |
| ----- | ----------------- |
| GRU   | 17.15 km          |
| LSTM  | 18.04 km          |
| TCN   | 15.74 km          |

---



---

## License

This project is intended for academic and research purposes.
