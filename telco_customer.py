# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve
import joblib
import warnings

warnings.filterwarnings("ignore")

# Step 1: Load and Preview Data
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully!\n")
    print("Dataset Preview:")
    print(df.head())
    return df

# Step 2: Data Overview
def data_overview(df):
    print("\nData Overview:")
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("Missing Values:\n", df.isnull().sum())
    print("Data Types:\n", df.dtypes)

# Step 3: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(df, target_column='churn'):
    print("\nPerforming Exploratory Data Analysis...")
    print("Churn Distribution:")
    sns.countplot(x=target_column, data=df)
    plt.show()

    # Display histograms of numerical columns
    df.hist(bins=20, figsize=(12, 8))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

# Step 4: Data Cleaning (Missing Values and Outliers)
def clean_data(df):
    print("\nCleaning data...")
    # Handle missing values
    df.fillna(df.median(), inplace=True)
    # Detect outliers using IQR and cap
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("Data cleaned successfully!\n")
    return df

# Step 5: Feature Engineering and Encoding
def feature_engineering(df, target_column='churn'):
    print("\nPerforming Feature Engineering...")
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=[1, 2, 3, 4, 5])

    # Encoding categorical features
    df = pd.get_dummies(df, drop_first=True)
    
    # Label encode target variable if necessary
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
    
    print("Feature engineering completed!\n")
    return df

# Step 6: Feature Scaling
def scale_features(df, target_column='churn'):
    print("\nScaling features...")
    scaler = StandardScaler()
    features = df.drop(columns=[target_column])
    df[features.columns] = scaler.fit_transform(features)
    print("Feature scaling done!\n")
    return df

# Step 7: Dimensionality Reduction
def dimensionality_reduction(df, target_column='churn', n_components=10):
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=n_components)
    features = df.drop(columns=[target_column])
    df_pca = pd.DataFrame(pca.fit_transform(features))
    df_pca[target_column] = df[target_column].values
    print("Dimensionality reduction completed!\n")
    return df_pca

# Step 8: Handling Imbalanced Data
def handle_imbalance(X, y):
    print("\nHandling data imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Data imbalance handled successfully!\n")
    return X_resampled, y_resampled

# Step 9: Model Building
def build_and_train_model(X_train, y_train):
    print("\nTraining the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed!\n")
    return model

# Step 10: Model Evaluation
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating the model...")
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Plot Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Step 11: Model Deployment (Save Model)
def save_model(model, filename='churn_model.pkl'):
    print("\nSaving the model...")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}\n")

# Step 12: Main Function to Execute All Steps
def main(filepath, target_column='churn'):
    # Step 1-2: Load and overview data
    df = load_data(filepath)
    data_overview(df)

    # Step 3: Perform EDA
    exploratory_data_analysis(df, target_column)

    # Step 4: Clean data
    df = clean_data(df)

    # Step 5: Feature Engineering
    df = feature_engineering(df, target_column)

    # Step 6: Feature Scaling
    df = scale_features(df, target_column)

    # Step 7: Dimensionality Reduction
    df = dimensionality_reduction(df, target_column, n_components=10)

    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 8: Handle Imbalance
    X_resampled, y_resampled = handle_imbalance(X, y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Step 9: Build and Train Model
    model = build_and_train_model(X_train, y_train)

    # Step 10: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Step 11: Save Model
    save_model(model)

# Run the script
if __name__ == "__main__":
    filepath = 'Telco_customer_churn.csv'  # Update this path to your dataset file
    target_column = 'churn'  # Update if the target column is named differently
    main(filepath, target_column)