# Hotel Bookings Analysis Repository

## Overview
This repository contains a Python-based application and scripts to analyze hotel booking datasets, visualize trends, and apply various preprocessing and statistical techniques.

### Repository Contents:
1. **app.py**: A Dash-based interactive web application for visualizing hotel bookings data.
   - Includes features such as outlier detection, PCA analysis, normality testing, and various graphical insights.
2. **project.py**: A Python script for data preprocessing, exploratory data analysis (EDA), and visualization of the hotel bookings dataset.
   - Implements advanced data cleaning, feature engineering, and statistical analysis.
3. **hotel_bookings.csv**: The primary dataset used in the analysis.
4. **hotel_pre.csv**: A preprocessed version of the dataset for quicker experimentation.
5. **requirements.txt**: A list of required Python libraries for running the application and scripts.

## Getting Started
Follow the steps below to clone the repository, set up the environment, and run the application or scripts.

### Prerequisites
Ensure that you have the following installed on your system:
- Python 3.8 or above
- pip (Python package installer)

### Cloning the Repository
1. Open your terminal or command prompt.
2. Run the following command to clone the repository:
   ```bash
   git clone <repository-url>
   ```
3. Navigate to the cloned repository:
   ```bash
   cd <repository-folder>
   ```

### Setting Up the Environment
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To launch the Dash web application for interactive data analysis:
1. Run the following command in the terminal:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:8033`.

### Running the Script
For preprocessing and exploratory analysis using the `project.py` script:
1. Run the script as follows:
   ```bash
   python project.py
   ```
2. Outputs include:
   - Visualizations of correlations, distributions, and trends.
   - Results of normality tests and PCA analysis.

## File Descriptions
### Python Files
- **app.py**:
  - A multi-tab interactive application with:
    - Outlier detection and visualization.
    - Normality tests (D'Agostino, Shapiro-Wilk, Kolmogorov-Smirnov).
    - PCA analysis for dimensionality reduction.
    - Choropleth maps and other advanced visualizations.

- **project.py**:
  - Key operations:
    - Data cleaning (handling missing values, outlier detection, feature engineering).
    - Statistical tests for normality.
    - Visualizations including heatmaps, box plots, pair plots, and 3D scatter plots.

### Datasets
- **hotel_bookings.csv**: Original dataset for analysis.
- **hotel_pre.csv**: Cleaned and preprocessed dataset.

### Dependencies
- Key libraries include:
  - Dash, Plotly, Pandas, Numpy, Seaborn, Scikit-learn, Matplotlib.
- Full list available in `requirements.txt`.

## Additional Notes
- The application listens on port 8033 by default. Ensure this port is not in use before running the app.
- Data visualizations are dynamically generated based on user inputs in the Dash app.

For any issues or contributions, feel free to open a pull request or contact the repository maintainer.

