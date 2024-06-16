## NBA Defensive Relationships

### Overview
This project investigates the relationship between the types of shots allowed by NBA teams and their defensive efficiency. By analyzing different defensive metrics, we aim to understand how specific defensive actions, such as allowing points in the paint and various types of 3-point shots, correlate with overall team defensive performance.

### Conclusions
- **Points in the Paint**: Teams that allow fewer points in the paint generally exhibit better defensive efficiency. This metric is a strong indicator of a team's ability to protect the rim and deter high-percentage shots close to the basket.
- **Open and Wide-Open 3-Pointers**: The number of open and wide-open 3-pointers allowed by a team also significantly impacts their defensive efficiency. Teams that can effectively contest or limit these shots tend to have better defensive ratings.
- **Combined Effect**: When considering points in the paint together with open and wide-open 3-pointers allowed, there is an even stronger correlation with defensive efficiency. Teams that excel in both areas tend to perform better defensively overall.

### Files and Descriptions

1. **Python Scripts** (located in `src/`):
    - `opponent_23.py`: Consolidates all provided opponent shooting statistics from the 2022-23 season into a single CSV file.
    - `opponent.py`: Consolidates all provided opponent shooting statistics from the 2023-24 season into a single CSV file.
    - `opponent_explore.py`: Conducts exploratory data analysis and feature engineering.

2. **CSV Files** (located in `data/`):
    - `opponent.csv`: Contains opponent statistics and actual defensive ratings for the 2023-24 season.
    - `opponent_2023.csv`: Contains opponent statistics and actual defensive ratings for the 2022-2023 season.

3. **Images** (located in `regression_results/`):
    - `pitp_regression.png`: Shows predicted vs. actual defensive ratings using opponent points in the paint.
    - `pitp_3s_regression.png`: Shows predicted vs. actual defensive ratings using points in the paint, wide-open 3-pointers allowed, and open 3-pointers allowed.
    - `pitp_3s_regression_23predictions.png`: Shows predicted vs. actual defensive ratings for the 2022-2023 season using the comprehensive regression model.

4. **Plots** (located in `plots/`):
    - Contains plots of the relationships between the various features, opponent points in the paint, wide open 3's allowed, and open 3's allowed, and defensive rating for the 2022-23 and 2023-24 seasons.

### Methodology

1. **Data Collection and Preprocessing**:
    - Data for opponent statistics and actual defensive ratings were collected and stored in CSV files.
    - The datasets were preprocessed to handle missing values, normalize features, and engineer relevant features for the regression models.

2. **Feature Engineering**:
    - Various features were considered and exploratory data analysis was conducted to understand the relationships between features and the target variable (defensive ratings).

3. **Model Training and Evaluation**:
    - Multiple regression models were trained using different combinations of features.
    - The models were evaluated based on mean absolute error and test scores to determine their accuracy.

4. **Predictions**:
    - The models were used to predict defensive ratings and the predictions were compared to actual defensive ratings to evaluate the models' performance.

### Results

- The model using opponent points in the paint achieved a mean absolute error of 1.840 and a score of 0.427 on the test set (2023-24 season).
- The model using opponent points in the paint, wide-open 3-pointers allowed, and open 3-pointers allowed achieved a mean absolute error of 1.805 and a score of 0.445 on the test set (2023-24 season).
- Predictions for the 2023-2024 season showed varying levels of accuracy, with some teams' predicted ratings closely matching their actual ratings.

### Usage
#### Scripts
1. **opponent_explore.py**
   This script conducts the data analysis and feature engineering.

   Example usage:
   ```bash
   cd src
   python3 opponent_explore.py
   ```
2. **opponent_23.py**
   This script consolidates all provided opponent shooting statistics from the 2022-23 season into a single CSV file.
   
   Example usage:
   ```bash
   cd src
   python3 opponent_23.py
   ```
2. **opponent.py**
   This script consolidates all provided opponent shooting statistics from the 2023-24 season into a single CSV file.
   
   Example usage:
   ```bash
   cd src
   python3 opponent.py
   ```
### Future Work
   - Incorporate additional seasons so the model generalizes better
