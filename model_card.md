# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model predicts binary output of a salary greater than or less than $50 thousand from input features.  It was trained using a Random Forest Classifier.

## Intended Use
This is a toy model for an academic demonstration.  It can be used for testing model pipeline methodoliges, or for other experimental scenarios.

## Training Data
The training data is the "Census Income" data set from UC Irvine Machine Learning Repository.  It includes demographic data and a "salary" feature.
- https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
25% of the data was selected at random from the "Census Income" data set for testing.  This is the default for the `train_test_split` function in scikit-learn.

## Metrics
The model was evaluated for precision, recall, and F1 score.  
#### Overall performance:
- Precision: 0.7378
- Recall: 0.6336
- F1: 0.6817

## Ethical Considerations
The features in the training data include age, marital status, race, and sex. Any model which is explicitly trained on these features may be unethical or even illegal to apply in many situations.  Such situations may include academic enrollment, employment, or financial loan applications.

## Caveats and Recommendations
Several feature slice F1 scores are 0%, due to small sample sizes mainly.  The biggest recommendation would be to use this model as a starting point for continued research and experimentation.  Try using a larger data set, or remove any features with under-represented slices, such as `native-country`.