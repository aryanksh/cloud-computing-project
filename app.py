import os
import csv
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import pyodbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import pickle
import io
from sklearn.preprocessing import MultiLabelBinarizer
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import traceback
import matplotlib.pyplot as plt

import matplotlib
app = Flask(__name__)

# Set the secret key needed for Flask session management
# IMPORTANT: In production, use a strong, persistent secret key,
# potentially loaded from environment variables or a config file.
app.secret_key = os.urandom(24)


# Models directory
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


# ========================
# Database Configuration
# ========================
CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:final-project-server2025.database.windows.net,1433;"
    "Database=retail-data-db;"
    "Uid=retail-admin;"
    "Pwd=cloud2025!;" # Consider using environment variables for credentials
    "Encrypt=yes;TrustServerCertificate=no;"
)

def get_connection():
    try:
        # Ensure timeout is sufficient for potential long queries
        return pyodbc.connect(CONN_STR, timeout=30)
    except pyodbc.Error as e:
        print(f"Database connection error: {str(e)}")
        # Log the error more formally if needed
        # logger.error(f"Database connection error: {str(e)}")
        raise # Re-raise the exception to be handled by the caller

# ========================
# Data Loading Functionality
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    # --- Authentication Check ---
    # Redirect to login if user is not logged in
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    # --- End Authentication Check ---

    if request.method == 'POST':
        conn = None # Initialize connection variable
        cursor = None # Initialize cursor variable
        try:
            # Get uploaded files
            transactions_file = request.files.get('transactions') # Use .get for safety
            households_file = request.files.get('households')
            products_file = request.files.get('products')

            # Check if all files were uploaded
            if not all([transactions_file, households_file, products_file]):
                 return jsonify({'error': 'Missing one or more required files (transactions, households, products)'}), 400

            conn = get_connection()
            cursor = conn.cursor()
            print("Database connection established for upload.")

            # --- Load Transactions ---
            print("Processing Transactions...")
            cursor.execute("TRUNCATE TABLE transactions")
            print("  Transactions table truncated.")
            # Use a context manager for reading file content
            with io.TextIOWrapper(transactions_file, encoding='utf-8') as text_file:
                transactions = list(csv.reader(text_file))
            if len(transactions) > 1: # Check if there are data rows
                # Basic validation: check if rows have expected number of columns (e.g., 9)
                valid_transactions = [row for row in transactions[1:] if len(row) == 9]
                if valid_transactions:
                    cursor.executemany(
                        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        valid_transactions
                    )
                    print(f"  Inserted {len(valid_transactions)} transaction records.")
                else:
                    print("  No valid transaction rows found after header/column check.")
            else:
                print("  Transactions file is empty or contains only a header.")
            conn.commit() # Commit after each table load

            # --- Load Households ---
            print("Processing Households...")
            cursor.execute("TRUNCATE TABLE household")
            print("  Household table truncated.")
            with io.TextIOWrapper(households_file, encoding='utf-8') as text_file:
                households = list(csv.reader(text_file))
            if len(households) > 1:
                valid_households = [row for row in households[1:] if len(row) == 9] # Assuming 9 columns
                if valid_households:
                    cursor.executemany(
                        "INSERT INTO household VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        valid_households
                    )
                    print(f"  Inserted {len(valid_households)} household records.")
                else:
                    print("  No valid household rows found after header/column check.")
            else:
                print("  Households file is empty or contains only a header.")
            conn.commit()

            # --- Load Products ---
            print("Processing Products...")
            cursor.execute("TRUNCATE TABLE products")
            print("  Products table truncated.")
            with io.TextIOWrapper(products_file, encoding='utf-8') as text_file:
                products = list(csv.reader(text_file))
            if len(products) > 1:
                valid_products = [row for row in products[1:] if len(row) == 3] # Assuming 3 columns
                if valid_products:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?)",
                        valid_products
                    )
                    print(f"  Inserted {len(valid_products)} product records.")
                else:
                     print("  No valid product rows found after header/column check.")
            else:
                 print("  Products file is empty or contains only a header.")
            conn.commit()

            print("Data loading process completed successfully.")
            return jsonify({'message': 'Data loaded successfully!'}), 200

        except pyodbc.Error as db_err:
            # Log the specific database error
            print(f"Database error during upload: {db_err}")
            # traceback.print_exc() # Optional: log full stack trace
            if conn: conn.rollback() # Rollback any partial changes
            return jsonify({'error': f'Database error occurred: {str(db_err)}'}), 500
        except Exception as e:
            # Log the general error
            print(f"An unexpected error occurred during upload: {e}")
            traceback.print_exc() # Log full stack trace for general errors
            if conn: conn.rollback()
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
        finally:
            # Ensure resources are closed even if errors occur
            if cursor:
                cursor.close()
                print("Database cursor closed.")
            if conn:
                conn.close()
                print("Database connection closed.")

    # Handle GET request
    return render_template('upload.html')


# ========================
# Data Analysis Functionality
# ========================
@app.route('/analysis', methods=['GET'])
def analysis():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    # --- End Authentication Check ---
    return render_template('analysis.html')

@app.route('/api/demographic-engagement', methods=['GET'])
def demographic_engagement():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query to analyze demographics impact on spending
        query = """
        SELECT
            h.INCOME_RANGE,
            h.HH_SIZE,
            CASE WHEN ISNULL(h.CHILDREN, 0) > 0 THEN 'With Children' ELSE 'No Children' END as CHILD_STATUS, -- Handle NULL Children
            h.L as LOCATION, -- Ensure 'L' is the correct column name for location
            COUNT(DISTINCT t.BASKET_NUM) as BASKET_COUNT,
            AVG(t.SPEND) as AVG_SPEND,
            SUM(t.SPEND) as TOTAL_SPEND
        FROM household h
        INNER JOIN transactions t ON h.HSHD_NUM = t.HSHD_NUM -- Use INNER JOIN if households must have transactions
        GROUP BY h.INCOME_RANGE, h.HH_SIZE, CASE WHEN ISNULL(h.CHILDREN, 0) > 0 THEN 'With Children' ELSE 'No Children' END, h.L
        ORDER BY TOTAL_SPEND DESC; -- Added semicolon
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify(results)
    except pyodbc.Error as db_err:
        print(f"Database error in demographic engagement: {db_err}")
        return jsonify({'error': f'Database error: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error in demographic engagement: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route('/api/engagement-over-time', methods=['GET'])
def engagement_over_time():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query for spending trends over time
        query = """
        SELECT
            t.YEAR,
            t.WEEK_NUM,
            p.DEPARTMENT,
            p.COMMODITY,
            SUM(t.SPEND) as TOTAL_SPEND,
            SUM(t.UNITS) as TOTAL_UNITS
        FROM transactions t
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM -- Use INNER JOIN if transactions must have products
        GROUP BY t.YEAR, t.WEEK_NUM, p.DEPARTMENT, p.COMMODITY
        ORDER BY t.YEAR, t.WEEK_NUM; -- Added semicolon
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify(results)
    except pyodbc.Error as db_err:
        print(f"Database error in engagement over time: {db_err}")
        return jsonify({'error': f'Database error: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error in engagement over time: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route('/api/basket-analysis', methods=['GET'])
def basket_analysis():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query to find frequent product combinations, limited in SQL
        # Using TOP 100 for SQL Server syntax
        query = """
        SELECT TOP 100
            t1.PRODUCT_NUM as PRODUCT1,
            t2.PRODUCT_NUM as PRODUCT2,
            p1.COMMODITY as COMMODITY1,
            p2.COMMODITY as COMMODITY2,
            COUNT(*) as FREQUENCY
        FROM transactions t1
        INNER JOIN transactions t2 ON t1.BASKET_NUM = t2.BASKET_NUM AND t1.HSHD_NUM = t2.HSHD_NUM AND t1.PRODUCT_NUM < t2.PRODUCT_NUM -- Ensure pairs are counted once
        INNER JOIN products p1 ON t1.PRODUCT_NUM = p1.PRODUCT_NUM
        INNER JOIN products p2 ON t2.PRODUCT_NUM = p2.PRODUCT_NUM
        GROUP BY t1.PRODUCT_NUM, t2.PRODUCT_NUM, p1.COMMODITY, p2.COMMODITY
        ORDER BY FREQUENCY DESC; -- Added semicolon
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        # Fetchall results already limited by SQL TOP 100
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify(results)
    except pyodbc.Error as db_err:
         print(f"Database error in basket analysis: {db_err}")
         return jsonify({'error': f'Database error: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error in basket analysis: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


# ========================
# Machine Learning Models
# =========================
@app.route('/models', methods=['GET'])
def models():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    # --- End Authentication Check ---
    return render_template('models.html')

@app.route('/api/churn-prediction', methods=['POST'])
def predict_churn():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    try:
        # Get data from request JSON payload
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must contain JSON data'}), 400

        # Define paths for model artifacts
        model_path = os.path.join(MODELS_DIR, 'churn_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'churn_scaler.pkl')
        features_path = os.path.join(MODELS_DIR, 'feature_cols.pkl') # Path for feature names

        # Check if model files exist
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            # Option 1: Return error (Safer - avoids potentially long training during request)
             return jsonify({'error': 'Churn model artifacts not found. Please train the model via the /api/train-models endpoint first.'}), 404
            # Option 2: Trigger training (Potentially slow - might cause request timeouts)
            # print("Churn model not found, attempting to train...")
            # train_churn_model() # This could take time
            # print("Training complete. Proceeding with prediction.")

        # Load model, scaler, and feature names
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        # Load the list of feature names the model expects
        with open(features_path, 'rb') as f:
             feature_cols = pickle.load(f)

        # Prepare input data: Extract features in the correct order using the loaded list
        # Use data.get(feature, 0) to handle missing features gracefully (defaulting to 0)
        try:
            input_values = [data.get(feature, 0) for feature in feature_cols]
        except TypeError: # Handle case where data is not a dictionary-like object
             return jsonify({'error': 'Invalid input data format. Expected a JSON object with feature keys.'}), 400

        input_data = np.array([input_values]) # Reshape for scaler/model (1 sample, N features)

        # Scale input data using the loaded scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0] # Get the prediction for the single sample
        # Get probability of the positive class (churn, assuming class 1 is churn)
        probability = model.predict_proba(input_scaled)[0][1]

        # Return prediction results
        return jsonify({
            'churn_prediction': int(prediction), # Convert numpy int to standard int
            'churn_probability': float(probability), # Convert numpy float to standard float
            'message': 'Churn predicted' if prediction == 1 else 'No churn predicted'
        })

    except FileNotFoundError:
        # This case should be caught by the initial check, but good practice to have
        print("Error: Model files not found during prediction.")
        return jsonify({'error': 'Model artifacts not found. Please train the model first.'}), 500
    except Exception as e:
        print(f"An error occurred during churn prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500


@app.route('/api/train-models', methods=['POST'])
def train_models():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    try:
        print("Starting model training process...")
        print("--- Training Churn Model ---")
        train_churn_model() # Call the churn training function
        print("--- Churn Model Training Complete ---")

        print("--- Training Basket Model ---")
        train_basket_model() # Call the basket training function
        print("--- Basket Model Training Complete ---")

        print("Model training process finished successfully.")
        return jsonify({'message': 'Models trained successfully!'})
    except Exception as e:
        print(f"Model training failed: {e}")
        traceback.print_exc()
        # Return a more specific error if possible (e.g., from the training functions)
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500


def train_churn_model():
    # This function encapsulates the logic for training the churn prediction model
    conn = None
    cursor = None
    try:
        print("Connecting to database for churn model training...")
        conn = get_connection()
        cursor = conn.cursor() # Although pandas reads SQL, keep cursor structure for consistency

        # --- Data Fetching ---
        print("Fetching required data from database...")
        # Fetch only necessary columns to optimize memory usage
        transactions_query = "SELECT HSHD_NUM, BASKET_NUM, PURCHASE_, SPEND FROM transactions"
        transactions = pd.read_sql(transactions_query, conn)

        # Get household data if needed for features (not used in current feature set, but potentially useful)
        # households_query = "SELECT HSHD_NUM, ... FROM household" # Select relevant columns
        # households = pd.read_sql(households_query, conn)

        print(f"Fetched {len(transactions)} transaction records.")

        # --- Data Preprocessing & Feature Engineering ---
        print("Preprocessing data and engineering features...")
        # Ensure PURCHASE_ is datetime type
        transactions['PURCHASE_'] = pd.to_datetime(transactions['PURCHASE_'], errors='coerce')
        # Drop rows where date conversion failed
        transactions.dropna(subset=['PURCHASE_'], inplace=True)

        if transactions.empty:
            raise ValueError("No valid transaction data available after date conversion.")

        # Determine the reference end date for time windows
        dataset_end_date = transactions['PURCHASE_'].max()
        if pd.isna(dataset_end_date):
             raise ValueError("Could not determine the maximum purchase date.")
        print(f"Dataset end date determined as: {dataset_end_date}")

        # Define date windows relative to the end date
        last_30_days_start = dataset_end_date - pd.Timedelta(days=29) # Include end date (30 days total)
        mid_period_end = last_30_days_start - pd.Timedelta(days=1)
        mid_period_start = dataset_end_date - pd.Timedelta(days=89) # 31-90 days prior

        # Filter data for recent 30 days
        recent_30 = transactions[transactions['PURCHASE_'] >= last_30_days_start]
        recent_agg = recent_30.groupby('HSHD_NUM').agg(
            spend_recent_30=('SPEND', 'sum'),
            transactions_recent_30=('BASKET_NUM', 'nunique')
        ).reset_index()

        # Filter data for 31-90 days prior
        mid_90 = transactions[(transactions['PURCHASE_'] >= mid_period_start) & (transactions['PURCHASE_'] <= mid_period_end)]
        mid_agg = mid_90.groupby('HSHD_NUM').agg(
            spend_mid_90=('SPEND', 'sum'),
            transactions_mid_90=('BASKET_NUM', 'nunique')
        ).reset_index()

        # Create a base dataframe with all unique household IDs
        all_hshd = pd.DataFrame(transactions['HSHD_NUM'].unique(), columns=['HSHD_NUM'])

        # Merge aggregated features onto the base dataframe
        behavior = all_hshd.merge(recent_agg, on='HSHD_NUM', how='left')
        behavior = behavior.merge(mid_agg, on='HSHD_NUM', how='left')

        # Fill NaNs with 0 for households with no activity in a period
        behavior = behavior.fillna(0)

        # Calculate derived behavioral features (ratios, drop rates, etc.)
        # Add a small epsilon to denominators to prevent division by zero
        epsilon = 1e-6
        behavior['spend_recent_ratio'] = behavior['spend_recent_30'] / (behavior['spend_mid_90'] + epsilon)
        behavior['transactions_recent_ratio'] = behavior['transactions_recent_30'] / (behavior['transactions_mid_90'] + epsilon)
        behavior['spend_drop_pct'] = (behavior['spend_mid_90'] - behavior['spend_recent_30']) / (behavior['spend_mid_90'] + epsilon)
        behavior['transactions_drop_pct'] = (behavior['transactions_mid_90'] - behavior['transactions_recent_30']) / (behavior['transactions_mid_90'] + epsilon)
        behavior['avg_spend_per_transaction_recent'] = behavior['spend_recent_30'] / (behavior['transactions_recent_30'] + epsilon)

        # Define the churn target variable
        # Churn = major spend drop (> 80%) OR no transactions in the most recent 30 days
        behavior['churn'] = np.where(
            (behavior['spend_drop_pct'] > 0.8) | (behavior['transactions_recent_30'] == 0),
            1,  # Churned
            0   # Not Churned
        )
        print(f"Churn definition applied. Churn rate: {behavior['churn'].mean():.2%}")

        # Define feature columns used for modeling
        feature_cols = [
            'spend_recent_30', 'spend_mid_90', 'spend_recent_ratio', 'spend_drop_pct',
            'transactions_recent_30', 'transactions_mid_90', 'transactions_recent_ratio',
            'transactions_drop_pct', 'avg_spend_per_transaction_recent'
        ]

        X = behavior[feature_cols].values
        y = behavior['churn'].values

        # Check if there are enough samples and both classes are present
        if len(X) < 10: # Arbitrary minimum threshold
            raise ValueError(f"Insufficient data for training ({len(X)} samples).")
        if len(np.unique(y)) < 2:
            print("Warning: Only one class present in the target variable. Model may not be effective.")
            # Decide how to handle: raise error, skip training, or proceed with caution
            # For now, proceed but be aware the model might predict only one class.

        # --- Model Training ---
        print("Splitting data into training and testing sets...")
        # Use stratify=y to maintain class proportions in splits, important for imbalanced datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y # Adjusted test_size to 30%
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        print("Standardizing features using StandardScaler...")
        scaler = StandardScaler()
        # Fit scaler ONLY on training data, then transform both train and test
        X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test) # We don't use X_test_scaled directly here, but it's good practice

        print("Training RandomForestClassifier model...")
        # Consider adding class_weight='balanced' if dataset is imbalanced
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        print("Model training complete.")

        # --- Save Artifacts ---
        print("Saving trained model, scaler, and feature list...")
        os.makedirs(MODELS_DIR, exist_ok=True) # Ensure directory exists
        joblib.dump(model, os.path.join(MODELS_DIR, 'churn_model.pkl'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'churn_scaler.pkl'))
        # Save the list of feature names used for training
        with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb') as f:
            pickle.dump(feature_cols, f)
        print("Churn model artifacts saved successfully.")

    except ValueError as ve:
        print(f"Data processing error during churn training: {ve}")
        # Re-raise to be caught by the calling function
        raise
    except pyodbc.Error as db_err:
        print(f"Database error during churn training: {db_err}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during churn model training: {e}")
        traceback.print_exc()
        raise
    finally:
        # Ensure database connection is closed
        if cursor: cursor.close()
        if conn: conn.close()
        print("Database connection closed for churn training.")

    # Removed return statement as artifacts are saved to files
    # return model, scaler, feature_cols


def train_basket_model():
    # This function trains a model for basket analysis/recommendations
    conn = None
    cursor = None
    try:
        print("Connecting to database for basket model training...")
        conn = get_connection()
        cursor = conn.cursor() # Keep cursor for consistency

        # --- Data Fetching ---
        print("Fetching transaction and product data...")
        # Join transactions with products to get commodity information directly
        query = """
        SELECT t.BASKET_NUM, p.COMMODITY
        FROM transactions t
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM;
        """
        basket_data = pd.read_sql(query, conn)
        print(f"Fetched {len(basket_data)} item records across baskets.")

        if basket_data.empty:
            raise ValueError("No transaction or product data found for basket analysis.")

        # Remove potential null/empty commodities
        basket_data.dropna(subset=['COMMODITY'], inplace=True)
        basket_data = basket_data[basket_data['COMMODITY'].str.strip() != '']

        # --- Data Preparation ---
        print("Grouping commodities by basket...")
        # Group by basket and create a list of unique commodities in each basket
        # Using set() within apply ensures uniqueness before converting to list
        basket_groups = basket_data.groupby('BASKET_NUM')['COMMODITY'].apply(lambda x: list(set(x))).reset_index()
        print(f"Processed {len(basket_groups)} unique baskets.")

        if basket_groups.empty:
             raise ValueError("No baskets found after grouping commodities.")

        # Optional Sampling: If memory is a concern with large datasets
        # Consider sampling a subset of baskets for training
        # sample_size = min(50000, len(basket_groups)) # Example sample size
        # if len(basket_groups) > sample_size:
        #     print(f"Sampling {sample_size} baskets for training...")
        #     basket_groups = basket_groups.sample(n=sample_size, random_state=42)

        # --- One-Hot Encode Baskets ---
        print("Applying MultiLabelBinarizer for one-hot encoding...")
        mlb = MultiLabelBinarizer()
        # Fit and transform the list of commodities per basket
        X_encoded = mlb.fit_transform(basket_groups['COMMODITY'])
        # Create a DataFrame from the encoded matrix with commodities as column names
        df_basket_ohe = pd.DataFrame(X_encoded, columns=mlb.classes_)
        print(f"Encoded baskets into a {df_basket_ohe.shape} matrix ({df_basket_ohe.shape[1]} unique commodities).")

        # --- Train Association Models (Simplified Feature Importance Approach) ---
        # This approach trains a classifier for each key item to see which *other* items
        # are important predictors (features) for its presence.
        print("Training association models (feature importance based)...")
        category_importances = {} # Dictionary to store importances for each modeled category

        # --- Determine Key Categories to Model ---
        # Example: Model categories appearing in at least 1% of sampled baskets or minimum 50 baskets
        min_frequency = max(50, int(0.01 * len(df_basket_ohe)))
        category_counts = df_basket_ohe.sum()
        key_categories = category_counts[category_counts >= min_frequency].index.tolist()

        if not key_categories:
            print("Warning: No categories met the frequency threshold. Using top 10 most frequent as fallback.")
            key_categories = category_counts.nlargest(10).index.tolist()

        if not key_categories:
            raise ValueError("No categories found to model for basket analysis.")

        print(f"Identified {len(key_categories)} key categories for modeling.")

        # --- Train a Model per Key Category ---
        for category in key_categories:
            print(f"  Training model for: {category}")
            # Target variable: 1 if the category is present in the basket, 0 otherwise
            y = df_basket_ohe[category].values
            # Feature set: All *other* categories' presence/absence
            X_features = df_basket_ohe.drop(columns=[category]).values
            # Get the names of the features (other categories)
            feature_names = df_basket_ohe.drop(columns=[category]).columns.tolist()

            # Check if the target variable has both classes (0 and 1)
            if len(np.unique(y)) < 2:
                print(f"  Skipping '{category}': Only one class (present/absent) found.")
                continue # Skip if the item is always or never present

            # Optional: Train/Test Split (can be skipped if goal is just importance on full data)
            # X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

            # Train a classifier (Random Forest used here)
            # Use fewer estimators and limit depth for faster training, as accuracy isn't the primary goal, but feature importance.
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
            model.fit(X_features, y) # Train on the full dataset (or X_train if split)

            # Store feature importances
            importances = model.feature_importances_
            # Create a dictionary mapping feature names (other categories) to their importance scores
            predictors_importance = dict(zip(feature_names, importances))
            # Sort by importance (descending) to easily find top associations
            sorted_importances = dict(sorted(predictors_importance.items(), key=lambda item: item[1], reverse=True))
            category_importances[category] = sorted_importances # Store the sorted importance dict

        # --- Save Artifacts ---
        print("Saving basket model artifacts (MLB and category importances)...")
        os.makedirs(MODELS_DIR, exist_ok=True) # Ensure directory exists
        # Save the MultiLabelBinarizer instance
        with open(os.path.join(MODELS_DIR, 'basket_mlb.pkl'), 'wb') as f:
            pickle.dump(mlb, f)
        # Save the dictionary containing feature importances for each key category
        # Renaming file to reflect content better
        with open(os.path.join(MODELS_DIR, 'basket_category_importances.pkl'), 'wb') as f:
            pickle.dump(category_importances, f)
        print("Basket model artifacts saved successfully.")

    except ValueError as ve:
        print(f"Data processing error during basket training: {ve}")
        raise
    except pyodbc.Error as db_err:
        print(f"Database error during basket training: {db_err}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during basket model training: {e}")
        traceback.print_exc()
        raise
    finally:
        # Ensure database connection is closed
        if cursor: cursor.close()
        if conn: conn.close()
        print("Database connection closed for basket training.")

    # Removed return statement as artifacts are saved to files
    # return category_importances


@app.route('/api/model-explanation', methods=['GET'])
def model_explanation():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---

    # Set Matplotlib backend to 'Agg' to prevent GUI issues in server environment
    # Must be done before importing pyplot or using plotting functions
    matplotlib.use('Agg')
    # Import pyplot after setting the backend
    import matplotlib.pyplot as plt

    # Use a BytesIO buffer to hold the generated image in memory
    img_buf = io.BytesIO()
    fig = None # Initialize fig to None

    try:
        # --- Get Request Parameters ---
        feature_idx_str = request.args.get('feature_idx')
        instance_idx_str = request.args.get('instance_idx')
        explanation_type = request.args.get('explanation_type', 'lime').lower() # Default to lime, lowercase

        # --- Validate Parameters ---
        try:
            # Default to index 0 if not provided
            feature_idx = int(feature_idx_str) if feature_idx_str is not None else 0
            instance_idx = int(instance_idx_str) if instance_idx_str is not None else 0
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid index parameter(s). Must be integers.'}), 400

        if explanation_type not in ['lime', 'pdp']:
             return jsonify({'error': 'Invalid explanation_type. Must be "lime" or "pdp".'}), 400

        # --- Load Required Model Artifacts ---
        print(f"Loading artifacts for {explanation_type.upper()} explanation...")
        model_path = os.path.join(MODELS_DIR, 'churn_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'churn_scaler.pkl')
        features_path = os.path.join(MODELS_DIR, 'feature_cols.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            return jsonify({'error': 'Model artifacts not found. Train the churn model first.'}), 404

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)
        print("Artifacts loaded.")

        # --- Regenerate Data for Explanation Context ---
        # This is potentially slow and resource-intensive.
        # In production, consider using a pre-saved, scaled test set or background dataset.
        print("Re-fetching and processing data for explanation context...")
        conn = None
        try:
            # (Simplified data fetching and processing - mirroring training logic)
            conn = get_connection()
            transactions_query = "SELECT HSHD_NUM, BASKET_NUM, PURCHASE_, SPEND FROM transactions"
            transactions = pd.read_sql(transactions_query, conn)
            transactions['PURCHASE_'] = pd.to_datetime(transactions['PURCHASE_'], errors='coerce')
            transactions.dropna(subset=['PURCHASE_'], inplace=True)
            if transactions.empty: raise ValueError("No valid transactions for explanation context.")

            dataset_end_date = transactions['PURCHASE_'].max()
            if pd.isna(dataset_end_date): raise ValueError("Cannot determine dataset end date for context.")

            last_30_days_start = dataset_end_date - pd.Timedelta(days=29)
            mid_period_end = last_30_days_start - pd.Timedelta(days=1)
            mid_period_start = dataset_end_date - pd.Timedelta(days=89)

            recent_30 = transactions[transactions['PURCHASE_'] >= last_30_days_start]
            recent_agg = recent_30.groupby('HSHD_NUM').agg(spend_recent_30=('SPEND', 'sum'), transactions_recent_30=('BASKET_NUM', 'nunique')).reset_index()
            mid_90 = transactions[(transactions['PURCHASE_'] >= mid_period_start) & (transactions['PURCHASE_'] <= mid_period_end)]
            mid_agg = mid_90.groupby('HSHD_NUM').agg(spend_mid_90=('SPEND', 'sum'), transactions_mid_90=('BASKET_NUM', 'nunique')).reset_index()

            all_hshd = pd.DataFrame(transactions['HSHD_NUM'].unique(), columns=['HSHD_NUM'])
            behavior = all_hshd.merge(recent_agg, on='HSHD_NUM', how='left').merge(mid_agg, on='HSHD_NUM', how='left').fillna(0)

            epsilon = 1e-6
            behavior['spend_recent_ratio'] = behavior['spend_recent_30'] / (behavior['spend_mid_90'] + epsilon)
            behavior['transactions_recent_ratio'] = behavior['transactions_recent_30'] / (behavior['transactions_mid_90'] + epsilon)
            behavior['spend_drop_pct'] = (behavior['spend_mid_90'] - behavior['spend_recent_30']) / (behavior['spend_mid_90'] + epsilon)
            behavior['transactions_drop_pct'] = (behavior['transactions_mid_90'] - behavior['transactions_recent_30']) / (behavior['transactions_mid_90'] + epsilon)
            behavior['avg_spend_per_transaction_recent'] = behavior['spend_recent_30'] / (behavior['transactions_recent_30'] + epsilon)
            behavior['churn'] = np.where((behavior['spend_drop_pct'] > 0.8) | (behavior['transactions_recent_30'] == 0), 1, 0)

            X = behavior[feature_cols].values
            y = behavior['churn'].values

            # Recreate the same train/test split as used in training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y # Match training split
            )

            # --- Scale data using the *loaded* scaler ---
            # Important: Use transform, not fit_transform, on the loaded scaler
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Explanation context data prepared and scaled.")

        except pyodbc.Error as db_err:
            print(f"Database error preparing explanation context: {db_err}")
            raise # Re-raise to be caught by the outer try-except
        except ValueError as ve:
            print(f"Data error preparing explanation context: {ve}")
            raise
        finally:
            if conn: conn.close()

        # --- Generate Explanation ---
        if explanation_type == 'lime':
            print(f"Generating LIME explanation for instance index: {instance_idx}")
            # Validate instance index against the test set size
            if not 0 <= instance_idx < len(X_test_scaled):
                return jsonify({'error': f'Instance index {instance_idx} is out of bounds for the test set (size {len(X_test_scaled)}).'}), 400

            sample_to_explain = X_test_scaled[instance_idx]

            # Initialize LIME explainer using the scaled training data as background distribution
            explainer = LimeTabularExplainer(
                training_data=X_train_scaled, # Background data for LIME perturbations
                feature_names=feature_cols,
                class_names=['Not Churn', 'Churn'], # Class names for interpretation
                mode='classification' # Specify classification task
            )

            # Generate the explanation for the selected instance
            exp = explainer.explain_instance(
                data_row=sample_to_explain,
                predict_fn=model.predict_proba, # Pass the model's probability prediction function
                num_features=len(feature_cols) # Show importance for all features
                # num_features=5 # Or limit to top N features
            )

            # Create plot from explanation object
            fig = exp.as_pyplot_figure()
            plt.tight_layout() # Adjust layout to prevent overlapping elements

            # Save plot to buffer
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0) # Rewind buffer to the beginning
            plt.close(fig) # Explicitly close the figure to free memory
            print("LIME plot generated.")
            return send_file(img_buf, mimetype='image/png')


        elif explanation_type == 'pdp':
            print(f"Generating PDP explanation for feature index: {feature_idx}")
            # Validate feature index
            if not 0 <= feature_idx < len(feature_cols):
                return jsonify({'error': f'Feature index {feature_idx} is out of bounds (0-{len(feature_cols)-1}).'}), 400

            target_feature_name = feature_cols[feature_idx]
            print(f"Target feature for PDP: {target_feature_name}")

            # Generate Partial Dependence Plot using scikit-learn's display
            # Use the scaled *training* data as the background for PDP calculation
            fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes explicitly
            try:
                display = PartialDependenceDisplay.from_estimator(
                    estimator=model,
                    X=X_train_scaled, # Use (scaled) training data
                    features=[feature_idx], # Index of the feature to plot
                    feature_names=feature_cols, # Pass all feature names
                    kind='average', # Plot average partial dependence ('average', 'individual', 'both')
                    # For classification, specify target class if needed (defaults often work)
                    # target = 1 # e.g., PDP for the 'Churn' class probability
                    ax=ax # Pass the created axes object
                )
                ax.set_title(f'Partial Dependence Plot for: {target_feature_name}')
                ax.set_ylabel('Partial Dependence (Churn Probability)') # Adjust label if needed
                plt.tight_layout()

                # Save plot to buffer
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                plt.close(fig) # Explicitly close the figure
                print("PDP plot generated.")
                return send_file(img_buf, mimetype='image/png')

            except Exception as pdp_err:
                 print(f"Error generating PDP plot: {pdp_err}")
                 traceback.print_exc()
                 # Ensure figure is closed even on error
                 if fig: plt.close(fig)
                 return jsonify({'error': f'Failed to generate PDP plot: {str(pdp_err)}'}), 500

        else:
            # This case should theoretically be caught by initial validation
            return jsonify({'error': 'Internal error: Invalid explanation type reached.'}), 500

    except FileNotFoundError:
        print("Error: Model files not found during explanation.")
        return jsonify({'error': 'Model artifacts not found. Train the model first.'}), 500
    except pyodbc.Error as db_err:
         print(f"Database error during explanation process: {db_err}")
         # Close figure if it exists and an error occurred before sending response
         if fig: plt.close(fig)
         return jsonify({'error': f'Database error during explanation: {str(db_err)}'}), 500
    except Exception as e:
        print(f"An unexpected error occurred during model explanation: {e}")
        traceback.print_exc()
        # Close figure if it exists and an error occurred before sending response
        if fig: plt.close(fig)
        # Create a more detailed error response for debugging if needed
        # traceback_str = traceback.format_exc()
        # return jsonify({'error': str(e), 'traceback': traceback_str}), 500
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


@app.route('/api/basket-recommendations', methods=['GET'])
def basket_recommendations():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    try:
        # Get basket items from query parameter 'items', split by comma
        basket_items_str = request.args.get('items', '')
        # Clean up input: remove empty strings resulting from trailing commas, etc.
        basket_items = [item.strip() for item in basket_items_str.split(',') if item.strip()]

        if not basket_items:
            return jsonify({'error': 'No valid items provided in the "items" query parameter.'}), 400

        print(f"Received basket items for recommendation: {basket_items}")

        # --- Load Basket Model Artifacts ---
        # Use the correct filenames saved during training
        mlb_path = os.path.join(MODELS_DIR, 'basket_mlb.pkl')
        importances_path = os.path.join(MODELS_DIR, 'basket_category_importances.pkl') # Corrected filename

        if not (os.path.exists(mlb_path) and os.path.exists(importances_path)):
             return jsonify({'error': 'Basket model artifacts not found. Train the models first.'}), 404

        # Load the MultiLabelBinarizer and the category importances dictionary
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
        with open(importances_path, 'rb') as f:
            category_importances = pickle.load(f) # This dict holds {category: {assoc_category: importance}}

        # --- Generate Recommendations ---
        # Use the loaded category_importances which store association strengths

        recommendations = {} # Use a dictionary to store potential recommendations and their scores

        # Iterate through items currently in the basket
        for item_in_basket in basket_items:
            # Check if we have an association model (importances) for this item
            if item_in_basket in category_importances:
                # Get the dictionary of associated items and their importance scores
                # These scores represent how strongly other items predict the presence of 'item_in_basket'
                associated_items_scores = category_importances[item_in_basket]

                # Iterate through the associated items and their scores
                for recommended_item, score in associated_items_scores.items():
                    # Rule: Recommend items NOT already in the basket
                    if recommended_item not in basket_items:
                        # Add or update the score for the potential recommendation
                        # If an item is recommended based on multiple basket items, sum or max its score?
                        # Using max score here: Keep the highest association score found so far.
                        current_score = recommendations.get(recommended_item, 0)
                        recommendations[recommended_item] = max(current_score, float(score))

        # --- Format and Sort Recommendations ---
        # Convert the recommendations dictionary to a list of dictionaries
        recommendation_list = [
            {'item': item, 'score': score} for item, score in recommendations.items()
        ]

        # Sort the recommendations by score in descending order
        sorted_recommendations = sorted(recommendation_list, key=lambda x: x['score'], reverse=True)

        # Limit the number of recommendations (e.g., top 5)
        top_recommendations = sorted_recommendations[:5]

        print(f"Generated recommendations: {top_recommendations}")

        return jsonify({
            'basket_items': basket_items, # Return the original basket for context
            'recommendations': top_recommendations
        })

    except FileNotFoundError:
         print("Error: Basket model files not found during recommendation.")
         return jsonify({'error': 'Basket model artifacts not found. Train the model first.'}), 500
    except Exception as e:
        print(f"An error occurred during basket recommendations: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


@app.route('/api/seasonal-trends', methods=['GET'])
def seasonal_trends():
    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query for seasonal patterns: Average spend/units per week for each commodity/department
        # Grouping by WEEK_NUM provides the seasonal component across years
        query = """
        SELECT
            t.WEEK_NUM,
            p.DEPARTMENT,
            p.COMMODITY,
            AVG(t.SPEND) as AVG_SPEND,
            AVG(t.UNITS) as AVG_UNITS
        FROM transactions t
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        GROUP BY t.WEEK_NUM, p.DEPARTMENT, p.COMMODITY
        ORDER BY t.WEEK_NUM, p.DEPARTMENT, p.COMMODITY; -- Order for clarity
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify(results)
    except pyodbc.Error as db_err:
        print(f"Database error in seasonal trends: {db_err}")
        return jsonify({'error': f'Database error: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error in seasonal trends: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route('/api/brand-preferences', methods=['GET'])
def brand_preferences():
    # Note: The current schema (transactions, products, household) doesn't seem
    # to have explicit 'Brand' information. This query analyzes preferences
    # at the DEPARTMENT and COMMODITY level instead.
    # If Brand data were available (e.g., in the products table), the query would change.

    # --- Authentication Check ---
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized access'}), 401
    # --- End Authentication Check ---
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query to analyze popularity (spend, unique customers) per Department/Commodity
        query = """
        SELECT
            p.DEPARTMENT,
            p.COMMODITY,
            SUM(t.SPEND) as TOTAL_SPEND,
            COUNT(DISTINCT t.HSHD_NUM) as UNIQUE_CUSTOMERS -- Count distinct households buying this
        FROM transactions t
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        -- No need to join household table just to count distinct HSHD_NUM from transactions
        -- JOIN household h ON t.HSHD_NUM = h.HSHD_NUM
        GROUP BY p.DEPARTMENT, p.COMMODITY
        ORDER BY TOTAL_SPEND DESC; -- Order by most popular based on spend
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify(results)
    except pyodbc.Error as db_err:
        print(f"Database error in brand preferences (commodity level): {db_err}")
        return jsonify({'error': f'Database error: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error in brand preferences (commodity level): {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


# ========================
# Interactive Search
# ========================
@app.route('/search', methods=['GET'])
def search():
    # --- Authentication Check ---
    # Decide if this needs authentication - is it sensitive?
    if not session.get('logged_in'):
       return jsonify({'error': 'Unauthorized access. Please log in.'}), 401
    # --- End Authentication Check ---

    # Get household number from query parameters
    hshd_num = request.args.get('hshd_num')
    if not hshd_num:
        return jsonify({'error': 'Missing required query parameter: "hshd_num"'}), 400

    # Optional: Validate hshd_num format if necessary (e.g., should be integer)
    # try:
    #     hshd_num_int = int(hshd_num)
    # except ValueError:
    #     return jsonify({'error': 'Invalid HSHD_NUM format. Must be an integer.'}), 400

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query to retrieve detailed transaction and household info for a specific household
        query = """
        SELECT
               h.HSHD_NUM, t.BASKET_NUM, t.DATE as TRANSACTION_DATE, t.PRODUCT_NUM,
               p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS,
               t.STORE_R, t.WEEK_NUM, t.YEAR,
               h.L as LOCATION, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE,
               h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
        FROM transactions t
        INNER JOIN household h ON t.HSHD_NUM = h.HSHD_NUM
        INNER JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE h.HSHD_NUM = ? -- Parameter placeholder for secure query execution
        ORDER BY t.DATE DESC, t.BASKET_NUM, p.DEPARTMENT, p.COMMODITY; -- Meaningful order
        """

        # Execute the query with the household number as a parameter
        cursor.execute(query, hshd_num) # Pass hshd_num safely

        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        if not results:
            # Return 200 OK with empty list or 404 Not Found?
            # 200 OK with empty list is often preferred for searches
            return jsonify([])
            # return jsonify({'message': f'No records found for HSHD_NUM {hshd_num}'}), 404

        return jsonify(results)

    except pyodbc.Error as db_err:
        print(f"Database error during search for HSHD_NUM {hshd_num}: {db_err}")
        return jsonify({'error': f'Database error occurred: {str(db_err)}'}), 500
    except Exception as e:
        print(f"Error during search for HSHD_NUM {hshd_num}: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# ========================
# Authentication Routes
# ========================

@app.route('/api/login', methods=['POST'])
def login():
    # Endpoint to handle login attempts via API (e.g., from JavaScript)
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Request body must contain JSON data'}), 400

        username = data.get('username')
        password = data.get('password')

        # --- Basic Credentials Check ---
        # Replace with a more secure method (e.g., hashed passwords from a database)
        # **NEVER store plain text passwords!**
        if username == 'admin' and password == 'password123': # Placeholder credentials
            # Set a flag in the session to indicate logged-in status
            session['logged_in'] = True
            # Optionally store username or user ID in session
            session['username'] = username
            print(f"User '{username}' logged in successfully.") # Log successful login
            return jsonify({'message': 'Logged in successfully!'}), 200
        else:
            print(f"Failed login attempt for username: '{username}'") # Log failed attempt
            return jsonify({'message': 'Invalid username or password'}), 401 # Use 401 Unauthorized

    except Exception as e:
        print(f"An error occurred during login: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500


@app.route('/login', methods=['GET'])
def login_page():
    # Renders the HTML login page
    # If already logged in, maybe redirect to index? (Optional)
    if session.get('logged_in'):
       return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    # Clears the session data for the user
    username = session.get('username', 'Unknown User') # Get username before popping
    session.pop('logged_in', None) # Remove the logged_in flag
    session.pop('username', None) # Remove username if stored
    # session.clear() # Alternatively, clear the entire session
    print(f"User '{username}' logged out.") # Log logout event
    # Redirect the user to the login page
    return redirect(url_for('login_page'))


 # ==================================
 # Home Page / Protected Route
 # ==================================
@app.route('/')
def index():
    # --- Authentication Check ---
    # Check if the 'logged_in' key exists and is True in the session
    if not session.get('logged_in'):
        # If not logged in, redirect to the login page
        return redirect(url_for('login_page'))
    # --- End Authentication Check ---

    # If logged in, render the main index page
    # Pass username to template if needed:
    # username = session.get('username', 'Guest')
    # return render_template('index.html', username=username)
    return render_template('index.html')


 # ==================================
 # Main Application Execution
 # ==================================
if __name__ == '__main__':
    # Set the secret key when running the app directly
    # app.secret_key is already set at the top level
    # No need to initialize app.config['logged_in'] anymore

    # Run the Flask development server
    # Add host='0.0.0.0' to make it accessible on the network
    # Add debug=True for development (auto-reloads, provides debugger)
    # IMPORTANT: Disable debug mode in production!
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0') # Example: run in debug mode, accessible on network
    # app.run() # Default run