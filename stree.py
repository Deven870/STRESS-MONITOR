import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\DEVENDER\Desktop\stress\StressLevelDataset.csv")  # change the filename to your actual path

# Step 1: Basic Info
print(df.head())
print(df.info())
print(df.isnull().sum())  # Check for missing values

# Step 2: Encode categorical columns (if any)
label_encoders = {}
categorical_columns = df.select_dtypes(include='object').columns
if not categorical_columns.empty:
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Step 3: Separate features and labels
X = df.drop('Stress Level', axis=1)  # Use the correct column name
y = df['Stress Level']

# Step 4: Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
