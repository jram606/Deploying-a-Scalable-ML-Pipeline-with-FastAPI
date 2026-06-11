# Model Card

## Model Details

This model is a supervised binary classification model that predicts whether an individual's income is greater than 50K or less than or equal to 50K. The model was trained using the provided Census dataset. The classifier used for this project is a Random Forest classifier from scikit-learn.

The model is part of a machine learning pipeline that includes data loading, preprocessing, model training, model evaluation, model serialization, slice-based performance evaluation, unit testing, and local deployment through a FastAPI application.

## Intended Use

This model is intended for educational use as part of the D501 machine learning deployment project. The purpose of the model is to demonstrate how to build a scalable machine learning pipeline, evaluate a classification model, test model-related code, and expose the trained model through a REST API.

This model should not be used for real-world decisions related to employment, lending, housing, benefits, eligibility, or any other decision that could affect a person's opportunities or access to resources.

## Training Data

The model was trained on the provided Census dataset located at `data/census.csv`. The target variable is `salary`, which classifies each record as either `>50K` or `<=50K`.

The data was split into a training dataset and a test dataset using an 80/20 train-test split. The split used a fixed random seed to make the results reproducible. Categorical features were processed using one-hot encoding. The target label was binarized before model training.

The categorical features used in the model were `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.

## Evaluation Data

The evaluation data was the held-out 20% test split from the provided Census dataset. This test data was not used to train the model. The same preprocessing approach used for the training data was applied to the test data, using the encoder fitted on the training data.

The model was evaluated on the overall test set and also on categorical slices of the test data. Slice-based evaluation was performed to review model performance for different values within categorical features.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Precision measures how many of the model's positive predictions were actually correct. Recall measures how many of the actual positive cases were correctly identified by the model. F1 score balances precision and recall into one metric.

The final model achieved the following performance on the test data:

- Precision: 0.7439
- Recall: 0.6186
- F1 score: 0.6755

These results show that the model performs moderately well overall, with stronger precision than recall. This means the model is more successful at limiting false positives than it is at identifying all positive cases.

## Ethical Considerations

The Census dataset contains demographic and socioeconomic features, including age, race, sex, education, marital status, occupation, and native country. These features may reflect historical inequality, social bias, and structural differences in income distribution.

Because of this, the model could learn patterns that are correlated with sensitive demographic characteristics. The predictions should be interpreted carefully, especially when reviewing performance across different demographic groups. This model should not be used for real-world decisions that affect individuals.

Slice-based performance evaluation was included to help identify whether the model performs differently across categorical groups. Additional fairness testing would be needed before considering this type of model for any production use.

## Caveats and Recommendations

This model was trained only on the provided Census dataset and may not generalize well to current populations, different geographic regions, or different time periods. The dataset may contain outdated patterns that do not fully represent modern income relationships.

The model's recall is lower than its precision, which means it may miss some individuals in the positive income class. Additional model tuning, feature engineering, fairness analysis, and validation on newer data would be recommended before any real-world use.

Future improvements could include comparing multiple algorithms, tuning hyperparameters, reviewing feature importance, measuring fairness metrics, and monitoring model performance over time.
