# Drug Allergy Prediction

## For Chulalongkorn AI Academy candidate screening

* Compatible with Python 3.6 and 3.7

## Problem introduction

* All doctors want to avoid prescribing drugs that maybe allergic to the patients
* Enzyme-linked immunospot (ELISpot) is a laboratory technique that tests whether the patient's immune cells will respond to particular drugs. This allows doctor to screen whether a drug is likely to be safe for the patient.
* As with any test, ELISpot is not perfect. Drugs that the patient is allergic to sometimes do not elicit any response in ELISpot test (False Negative) and vice versa.
* Hence, we want to develop a prediction model for drug allergy based on patient information, drug information, and ELISpot result.

## Getting Started

Run the following command in your Terminal to install Project Dependencies

```sh
python install -r requirements.txt
```

To begin training the project, download and rename the dataset from the Kaggle page to the data/folder as so:

> data/drugs.csv  

Create a Drug Allergy Estimator using the following command:

```sh
python drugs.py
```

### Code Walkthrough

You can follow along the attached Jupyter Notebook titled `drug_allergy.ipynb` on this project and see how data was treated before a Classification Model was trained on the data.
Rule #1 of Data Science is Garbage In, Garbage Out. This means that we need to identify how to work with the data given first. 

Duplicate Rows were dropped before training.

```python
data.drop_duplicates(inplace=True)
```

Also there were a lot of null data given when it came to Underlying Conditions, these rows were imputed using the Population Mode of each of these features.
The assumption here is that, if the data for this was not recorded for the individual, it is not an outstanding underlying condition for said individual.

```python
data.Underlying_Condition_A.fillna(data.Underlying_Condition_A.mode().iloc[0], inplace=True)  
data.Underlying_Condition_D.fillna(data.Underlying_Condition_D.mode().iloc[0], inplace=True)  
data.Underlying_Condition_E.fillna(data.Underlying_Condition_E.mode().iloc[0], inplace=True)  
```

Ordinal and Categorical Features were flattened into different columns to remove each category's amptitudal impact on each other.

```python
for column in data.columns:  
  matched = re.match(r'\w+_Group', column)  
  if matched is not None:  
    data = data.join(get_dummies(data[column], prefix=column, dummy_na=True))  
    data.drop([column], axis=1, inplace=True)  
```

Suspicion Score for Suspected drug and near-suspected drug were combined into one row when categorically expanded as it is much different from the negatively controlled drug.\

The final result of the prediction model performs really well on training data.
```sh
[INFO] Number of Wrong Predictions: 4 / 22
[INFO] R2: 0.0833
[INFO] Mean Squared Error: 0.1818
[INFO] Root Mean Squared Error: 0.4264
```