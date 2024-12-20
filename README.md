# Project Name

This project leverages Python for advanced data analysis and machine learning workflows using libraries such as NumPy, PySpark, Scikit-learn, and XGBoost. Optimized for Python versions >= 3.7 and < 3.10, it utilizes PySpark for scalable processing of the Yelp Challenge Dataset, enabling accurate predictions of business ratings based on user data.

---

## Model Details

To enhance predictive accuracy and reduce RMSE, the recommendation system was transitioned to an XGBoost model that utilizes comprehensive feature engineering. This robust model integrates a diverse range of features derived from user metadata, business characteristics, geographic data, and social interactions, enabling a more sophisticated understanding of patterns and behaviors.

### Key Features:

1. **User Metadata**: Incorporates user-specific attributes to personalize predictions.
2. **Business Characteristics**: Leverages information such as business categories to provide contextual insights.
3. **Geographic Data**: Employs clustering of businesses based on location and analyzes regional trends.
4. **Interaction Features**: Captures nuanced relationships through user-business rating patterns and geographic influences.

### Methodology:
- **Feature Scaling**: Applied `MinMaxScaler` to standardize feature values.
- **Hyperparameter Tuning**: Optimized XGBoost parameters and implemented early stopping to mitigate overfitting.
- **Efficient Training**: Employed the `tree-method: hist` in XGBoost, which utilizes a histogram-based algorithm for accelerated computation without compromising accuracy.

These enhancements resulted in a significant reduction in RMSE, achieving a score of approximately **0.97768**, underscoring the model's improved performance.

---

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

---

## Requirements

### Python Version

- Python >= 3.7, < 3.10

### Libraries

The following libraries are required:

- `numpy==2.2.0`
- `pyspark==3.5.4`
- `scikit-learn==1.6.0`
- `xgboost==0.72.1`

---

## Setup

### Using Conda

1. Create a new Conda environment:
   ```bash
   conda create --name myenv python=3.9
   conda activate myenv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using Virtualenv

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate   # For Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Download PySpark

1. Download PySpark from the [official Apache Spark website](https://spark.apache.org/downloads.html) (v3.5.4).
2. Extract the downloaded archive and place it in the same folder as the repository.
3. Ensure the path to PySpark is correctly set up in your environment.

### Configure Environment Variables

After cloning the repository, set up the following environment variables to ensure proper execution:

1. Export the `JAVA_HOME` variable:

   ```bash
   export JAVA_HOME=/path/to/java/jdk
   ```

   Example: `/Library/Java/JavaVirtualMachines/jdk1.8.0_202.jdk/Contents/Home`

2. Export the `SPARK_HOME` variable:

   ```bash
   export SPARK_HOME=/path/to/spark
   ```

   Example: `~/Python_Projects/your_repo_folder/spark-3.1.2-bin-hadoop3.2`

3. Update the `PATH`:

   ```bash
   export PATH=$SPARK_HOME/bin:$PATH
   ```

For convenience, you can add these exports to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`).

### Check and Install Java

1. Verify your current Java version:

   ```bash
   java -version
   ```

   Ensure that the version is compatible with Spark (e.g., Java 8 or 11).

2. If Java is not installed or the version is incompatible, download and install it:

   - From the [Oracle Java website](https://www.oracle.com/java/technologies/javase-downloads.html) or install OpenJDK:
     ```bash
     sudo apt install openjdk-8-jdk  # For Ubuntu/Debian
     brew install openjdk@8         # For macOS
     ```

3. Verify the installation:

   ```bash
   java -version
   ```

---

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Run the main script:

   ```bash
   ./spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G xgboost_rec_yelp.py <folder_path> <test_file_name> <output_file_name>
   ```

  Param: folder_path: the path of dataset folder (yelp_challenge_dataset)
  Param: test_file_name: the name of the testing file (e.g., yelp_val.csv), including the file path
  Param: output_file_name: the name of the prediction result file in csv form, including the file path
   
---

## Features

- Data analysis using NumPy and PySpark
- Machine learning models with Scikit-learn and XGBoost
- Scalable processing with PySpark

---

## License

This project is licensed under the [MIT License](LICENSE).

---
