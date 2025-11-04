import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

class DiabetesDataCleaner:
    def __init__(self, file_path):
        """Load dataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        self.df = pd.read_csv(file_path)
        self.original_df = self.df.copy()
        print(f"Dataset loaded successfully with shape: {self.df.shape}")

        # Default bin settings
        self.age_bins = [20, 35, 55, 100]
        self.bmi_bins = [0, 18.5, 25, 30, 100]
        self.glucose_bins = [0, 99, 126, 500]

    def explore_data(self):
        print("\n--- PHASE 1: Data Exploration ---")
        print(self.df.info())
        print(self.df.isnull().sum())

        print("\nDescriptive Statistics:\n", self.df.describe().T)        

        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=self.df)
        plt.title('Target Class Distribution')
        plt.show()

        critical_features = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']
        print("\nZeros in Critical Features:")
        for col in critical_features:
            count = (self.df[col] == 0).sum()
            print(f"{col}: {count} zeros ({round(count/len(self.df)*100,2)}%)")

        plt.figure(figsize=(15,10))
        self.df[critical_features].hist()
        plt.suptitle("Critical Feature Distributions")
        plt.show()

        msno.matrix(self.df)
        plt.show()
        return self.df

    def clean_data(self):
        print("\n--- PHASE 2: Data Cleaning ---")
        critical_features = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']
        
        # Replace zeros with NaN
        self.df[critical_features] = self.df[critical_features].replace(0, np.nan)

        # Visualize missing values
        plt.figure(figsize=(10, 5))
        msno.matrix(self.df)
        plt.title("Missing Values After Replacing Zeros")
        plt.show()

        # Median imputation
        imputer = SimpleImputer(strategy="median")
        self.df[critical_features] = imputer.fit_transform(self.df[critical_features])
        print("Median imputation applied.")

        # Remove duplicates
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.df)} duplicate rows.")

        # Visualize distributions before outlier capping
        numeric_cols = ['Glucose','Diastolic_BP','Skin_Fold','Serum_Insulin','BMI','Diabetes_Pedigree','Age']
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(1, len(numeric_cols), i)
            sns.boxplot(y=self.df[col])
            plt.title(f'{col} (Before Capping)')
        plt.tight_layout()
        plt.show()

        # Outlier treatment using IQR
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            self.df[col] = np.clip(self.df[col], lower, upper)
        print("Outliers capped using IQR method.")

        # Visualize distributions after outlier capping
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(1, len(numeric_cols), i)
            sns.boxplot(y=self.df[col])
            plt.title(f'{col} (After Capping)')
        plt.tight_layout()
        plt.show()

        return self.df

    def transform_data(self):
        print("\n--- PHASE 3: Data Transformation ---")
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=self.age_bins, labels=["Young","Middle-aged","Senior"])
        self.df['BMICategory'] = pd.cut(self.df['BMI'], bins=self.bmi_bins, labels=["Underweight","Normal","Overweight","Obese"])
        self.df['GlucoseCategory'] = pd.cut(self.df['Glucose'], bins=self.glucose_bins, labels=["Normal","Prediabetes","Diabetes"])

        self.df = pd.get_dummies(self.df, columns=['AgeGroup','BMICategory','GlucoseCategory'], drop_first=True, dtype=int)

        numeric_cols = ['Pregnant','Glucose','Diastolic_BP','Skin_Fold','Serum_Insulin','BMI','Diabetes_Pedigree','Age']
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        print("StandardScaler applied.")
        return self.df

    def reduce_data(self, k_features=10):
        print("\n--- PHASE 4: Feature Selection & Dimensionality Reduction ---")
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        print("Selected Features:", list(selected_features))
        return self.df[selected_features], y

    def balance_data(self, X, y):
        print("\n--- PHASE 5: Data Balancing Using SMOTE ---")
        
        # Before balancing visualization
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y)
        plt.title("Class Distribution Before SMOTE")
        plt.show()
        print("Before Balancing:", Counter(y))
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # After balancing visualization
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y_res)
        plt.title("Class Distribution After SMOTE")
        plt.show()
        print("After Balancing:", Counter(y_res))
        
        # Combine into a DataFrame
        df_balanced = pd.DataFrame(X_res, columns=X.columns)
        df_balanced['Class'] = y_res
        
        return df_balanced


    def save_clean_data(self, df=None, path="../data/cleaned/Diabetes_Cleaned.csv"):
        if df is None:
            df = self.df
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Final cleaned dataset saved successfully to {path}")
