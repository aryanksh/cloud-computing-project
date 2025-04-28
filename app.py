import os
import csv
from flask import Flask, render_template, request, jsonify
import pyodbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import pickle
from flask import send_file
import io
from sklearn.preprocessing import MultiLabelBinarizer
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import traceback

app = Flask(__name__)


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
    "Pwd=cloud2025!;"
    "Encrypt=yes;TrustServerCertificate=no;"
)

def get_connection():
    try:
        return pyodbc.connect(CONN_STR)
    except pyodbc.Error as e:
        print(f"Database connection error: {str(e)}")
        raise

# ========================
# Data Loading Functionality
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        try:
            # Get uploaded files
            transactions_file = request.files['transactions']
            households_file = request.files['households']
            products_file = request.files['products']

            conn = get_connection()
            cursor = conn.cursor()

            # Load Transactions
            cursor.execute("TRUNCATE TABLE transactions")
            transactions = list(csv.reader(transactions_file.read().decode('utf-8').splitlines()))
            cursor.executemany(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                transactions[1:]  # Skip header
            )

            # Load Households
            cursor.execute("TRUNCATE TABLE household")
            households = list(csv.reader(households_file.read().decode('utf-8').splitlines()))
            cursor.executemany(
                "INSERT INTO household VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                households[1:]
            )

            # Load Products
            cursor.execute("TRUNCATE TABLE products")
            products = list(csv.reader(products_file.read().decode('utf-8').splitlines()))
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?)",
                products[1:]
            )

            conn.commit()
            return jsonify({'message': 'Data loaded successfully!'}), 200

        except Exception as e:
            conn.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            cursor.close()
            conn.close()

    return render_template('upload.html')

# ========================
# Data Analysis Functionality
# ========================
@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html')

@app.route('/api/demographic-engagement', methods=['GET'])
def demographic_engagement():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query to analyze demographics impact on spending
        query = """
        SELECT 
            h.INCOME_RANGE, 
            h.HH_SIZE, 
            CASE WHEN h.CHILDREN > 0 THEN 'With Children' ELSE 'No Children' END as CHILD_STATUS,
            h.L as LOCATION,
            COUNT(DISTINCT t.BASKET_NUM) as BASKET_COUNT,
            AVG(t.SPEND) as AVG_SPEND,
            SUM(t.SPEND) as TOTAL_SPEND
        FROM household h
        JOIN transactions t ON h.HSHD_NUM = t.HSHD_NUM
        GROUP BY h.INCOME_RANGE, h.HH_SIZE, CASE WHEN h.CHILDREN > 0 THEN 'With Children' ELSE 'No Children' END, h.L
        ORDER BY TOTAL_SPEND DESC
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/engagement-over-time', methods=['GET'])
def engagement_over_time():
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
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        GROUP BY t.YEAR, t.WEEK_NUM, p.DEPARTMENT, p.COMMODITY
        ORDER BY t.YEAR, t.WEEK_NUM
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/basket-analysis', methods=['GET'])
def basket_analysis():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query to find frequent product combinations
        query = """
        SELECT 
            t1.PRODUCT_NUM as PRODUCT1,
            t2.PRODUCT_NUM as PRODUCT2,
            p1.COMMODITY as COMMODITY1,
            p2.COMMODITY as COMMODITY2,
            COUNT(*) as FREQUENCY
        FROM transactions t1
        JOIN transactions t2 ON t1.BASKET_NUM = t2.BASKET_NUM AND t1.HSHD_NUM = t2.HSHD_NUM AND t1.PRODUCT_NUM < t2.PRODUCT_NUM
        JOIN products p1 ON t1.PRODUCT_NUM = p1.PRODUCT_NUM
        JOIN products p2 ON t2.PRODUCT_NUM = p2.PRODUCT_NUM
        GROUP BY t1.PRODUCT_NUM, t2.PRODUCT_NUM, p1.COMMODITY, p2.COMMODITY
        ORDER BY FREQUENCY DESC
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()[:100]]  # Limit to top 100
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# ========================
# Machine Learning Models
# =========================
@app.route('/models', methods=['GET'])
def models():
    return render_template('models.html')
@app.route('/api/churn-prediction', methods=['POST'])
def predict_churn():
    try:
        # Get data from request
        data = request.get_json()
        
        # Load model and scaler
        model_path = os.path.join(MODELS_DIR, 'churn_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'churn_scaler.pkl')
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            # Train the model if it doesn't exist
            train_churn_model()
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Prepare input data
        input_data = np.array([
            [data.get('spend_recent_30', 0), 
                data.get('spend_mid_90', 0),
                data.get('spend_recent_ratio', 0),
                data.get('spend_drop_pct', 0),
                data.get('transactions_recent_30', 0),
                data.get('transactions_mid_90', 0),
                data.get('transactions_recent_ratio', 0),
                data.get('transactions_drop_pct', 0),
                data.get('avg_spend_per_transaction_recent', 0)]
        ])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'message': 'Churn predicted' if prediction == 1 else 'No churn predicted'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    try:
        train_churn_model()
        train_basket_model()
        return jsonify({'message': 'Models trained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def train_churn_model():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get transactions data
    transactions_query = """
    SELECT * FROM transactions
    """
    cursor.execute(transactions_query)
    transactions_rows = cursor.fetchall()
    transactions_cols = [column[0] for column in cursor.description]
    transactions = pd.DataFrame.from_records(transactions_rows, columns=transactions_cols)
    
    # Get households data
    households_query = """
    SELECT * FROM household
    """
    cursor.execute(households_query)
    households_rows = cursor.fetchall()
    households_cols = [column[0] for column in cursor.description]
    households = pd.DataFrame.from_records(households_rows, columns=households_cols)
    
    # Get products data
    products_query = """
    SELECT * FROM products
    """
    cursor.execute(products_query)
    products_rows = cursor.fetchall()
    products_cols = [column[0] for column in cursor.description]
    products = pd.DataFrame.from_records(products_rows, columns=products_cols)
    
    # Clean column names
    transactions.columns = transactions.columns.str.strip()
    households.columns = households.columns.str.strip()
    products.columns = products.columns.str.strip()
    
    # Convert date
    transactions['PURCHASE_'] = transactions['DATE']
    transactions['PURCHASE_'] = pd.to_datetime(transactions['PURCHASE_'], errors='coerce')
    
    # Merge datasets
    transactions_households = transactions.merge(households, on='HSHD_NUM', how='left')
    full_data = transactions_households.merge(products, on='PRODUCT_NUM', how='left')
    
    # Dataset reference end date - use max date from transactions
    dataset_end_date = transactions['PURCHASE_'].max()
    
    # Define date windows
    last_30_days = dataset_end_date - pd.Timedelta(days=30)
    days_31_to_90 = dataset_end_date - pd.Timedelta(days=90)
    
    # Recent 30-day spend and transactions
    recent_30 = full_data[(full_data['PURCHASE_'] >= last_30_days) & (full_data['PURCHASE_'] <= dataset_end_date)]
    recent_agg = recent_30.groupby('HSHD_NUM').agg(
        spend_recent_30=('SPEND', 'sum'),
        transactions_recent_30=('BASKET_NUM', 'nunique')
    ).reset_index()
    
    # 31–90 day spend and transactions
    mid_90 = full_data[(full_data['PURCHASE_'] >= days_31_to_90) & (full_data['PURCHASE_'] < last_30_days)]
    mid_agg = mid_90.groupby('HSHD_NUM').agg(
        spend_mid_90=('SPEND', 'sum'),
        transactions_mid_90=('BASKET_NUM', 'nunique')
    ).reset_index()
    
    # Lifetime aggregates
    lifetime_agg = full_data.groupby('HSHD_NUM').agg(
        spend_lifetime=('SPEND', 'sum'),
        transactions_lifetime=('BASKET_NUM', 'nunique'),
        first_purchase=('PURCHASE_', 'min'),
        last_purchase=('PURCHASE_', 'max')
    ).reset_index()
    
    # Merge all
    behavior = lifetime_agg.merge(recent_agg, on='HSHD_NUM', how='left')
    behavior = behavior.merge(mid_agg, on='HSHD_NUM', how='left')
    
    # Fill missing values
    behavior = behavior.fillna(0)
    
    # Days active
    behavior['days_active'] = (behavior['last_purchase'] - behavior['first_purchase']).dt.days
    
    # Recent engagement ratios
    behavior['spend_recent_ratio'] = behavior['spend_recent_30'] / (behavior['spend_mid_90'] + 1)
    behavior['transactions_recent_ratio'] = behavior['transactions_recent_30'] / (behavior['transactions_mid_90'] + 1)
    
    # Drop rates
    behavior['spend_drop_pct'] = (behavior['spend_mid_90'] - behavior['spend_recent_30']) / (behavior['spend_mid_90'] + 1)
    behavior['transactions_drop_pct'] = (behavior['transactions_mid_90'] - behavior['transactions_recent_30']) / (behavior['transactions_mid_90'] + 1)
    
    # Recent monetary per transaction
    behavior['avg_spend_per_transaction_recent'] = behavior['spend_recent_30'] / (behavior['transactions_recent_30'] + 1)
    
    # Churn = major drop (> 80% spend drop) OR no recent activity
    behavior['churn'] = np.where(
        (behavior['spend_drop_pct'] > 0.8) | (behavior['transactions_recent_30'] == 0),
        1, 0
    )
    
    # Feature Selection
    feature_cols = [
        'spend_recent_30', 'spend_mid_90', 'spend_recent_ratio', 'spend_drop_pct',
        'transactions_recent_30', 'transactions_mid_90', 'transactions_recent_ratio',
        'transactions_drop_pct', 'avg_spend_per_transaction_recent'
    ]
    
    X = behavior[feature_cols].values
    y = behavior['churn'].values
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, 'churn_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'churn_scaler.pkl'))
    pickle.dump(feature_cols, open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb'))
    
    cursor.close()
    conn.close()
    
    return model, scaler, feature_cols

def train_basket_model():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get transactions data
    transactions_query = """
    SELECT * FROM transactions
    """
    cursor.execute(transactions_query)
    transactions_rows = cursor.fetchall()
    transactions_cols = [column[0] for column in cursor.description]
    transactions = pd.DataFrame.from_records(transactions_rows, columns=transactions_cols)
    
    # Get products data
    products_query = """
    SELECT * FROM products
    """
    cursor.execute(products_query)
    products_rows = cursor.fetchall()
    products_cols = [column[0] for column in cursor.description]
    products = pd.DataFrame.from_records(products_rows, columns=products_cols)
    
    # Clean column names
    transactions.columns = transactions.columns.str.strip()
    products.columns = products.columns.str.strip()
    
    # Map product to commodity
    product_to_commodity = products.set_index('PRODUCT_NUM')['COMMODITY'].to_dict()
    transactions['COMMODITY'] = transactions['PRODUCT_NUM'].map(product_to_commodity)
    
    # Sample baskets to avoid memory issues
    sampled_baskets = transactions['BASKET_NUM'].drop_duplicates().sample(n=min(2000, len(transactions['BASKET_NUM'].unique())), random_state=42)
    sampled_transactions = transactions[transactions['BASKET_NUM'].isin(sampled_baskets)]
    
    # Group baskets by BASKET_NUM and collect all COMMODITYs
    basket_categories = sampled_transactions.groupby('BASKET_NUM')['COMMODITY'].apply(list).reset_index()
    
    # One-hot encode the baskets
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(basket_categories['COMMODITY'])
    df_basket_ohe = pd.DataFrame(X, columns=mlb.classes_)
    
    # Train basket association model for a few key categories
    category_importances = {}
    key_categories = df_basket_ohe.columns[:10]  # Take first 10 categories
    
    for category in key_categories:
        # Prepare target: 1 if category present, 0 otherwise
        y = df_basket_ohe[category].values
        X_features = df_basket_ohe.drop(columns=[category]).values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_filename = f'basket_model_{category.replace(" ", "_")}.pkl'
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
        
        # Store feature importances
        importances = model.feature_importances_
        predictors = df_basket_ohe.drop(columns=[category]).columns
        category_importances[category] = dict(zip(predictors, importances))
    
    # Save category importances and mlb
    pickle.dump(category_importances, open(os.path.join(MODELS_DIR, 'category_importances.pkl'), 'wb'))
    pickle.dump(mlb, open(os.path.join(MODELS_DIR, 'mlb.pkl'), 'wb'))
    
    cursor.close()
    conn.close()
    
    return category_importances

@app.route('/api/model-explanation', methods=['GET'])
def model_explanation():
    try:
        # Get parameters
        feature_idx = int(request.args.get('feature_idx', 0))
        instance_idx = int(request.args.get('instance_idx', 0))
        explanation_type = request.args.get('explanation_type', 'lime')
        
        # Connect to database and get transactions data
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get transactions data
        transactions_query = """
        SELECT * FROM transactions
        """
        cursor.execute(transactions_query)
        transactions_rows = cursor.fetchall()
        transactions_cols = [column[0] for column in cursor.description]
        transactions = pd.DataFrame.from_records(transactions_rows, columns=transactions_cols)
        
        # Get households data
        households_query = """
        SELECT * FROM household
        """
        cursor.execute(households_query)
        households_rows = cursor.fetchall()
        households_cols = [column[0] for column in cursor.description]
        households = pd.DataFrame.from_records(households_rows, columns=households_cols)
        
        # Get products data
        products_query = """
        SELECT * FROM products
        """
        cursor.execute(products_query)
        products_rows = cursor.fetchall()
        products_cols = [column[0] for column in cursor.description]
        products = pd.DataFrame.from_records(products_rows, columns=products_cols)
        
        cursor.close()
        conn.close()
        
        # Clean column names
        transactions.columns = transactions.columns.str.strip()
        households.columns = households.columns.str.strip()
        products.columns = products.columns.str.strip()
        
        # Convert date
        transactions['PURCHASE_'] = transactions['DATE']
        transactions['PURCHASE_'] = pd.to_datetime(transactions['PURCHASE_'], errors='coerce')
        
        # Merge datasets
        transactions_households = transactions.merge(households, on='HSHD_NUM', how='left')
        full_data = transactions_households.merge(products, on='PRODUCT_NUM', how='left')
        
        # Dataset reference end date - use max date from transactions
        dataset_end_date = transactions['PURCHASE_'].max()
        
        # Define date windows
        last_30_days = dataset_end_date - pd.Timedelta(days=30)
        days_31_to_90 = dataset_end_date - pd.Timedelta(days=90)
        
        # Recent 30-day spend and transactions
        recent_30 = full_data[(full_data['PURCHASE_'] >= last_30_days) & (full_data['PURCHASE_'] <= dataset_end_date)]
        recent_agg = recent_30.groupby('HSHD_NUM').agg(
            spend_recent_30=('SPEND', 'sum'),
            transactions_recent_30=('BASKET_NUM', 'nunique')
        ).reset_index()
        
        # 31–90 day spend and transactions
        mid_90 = full_data[(full_data['PURCHASE_'] >= days_31_to_90) & (full_data['PURCHASE_'] < last_30_days)]
        mid_agg = mid_90.groupby('HSHD_NUM').agg(
            spend_mid_90=('SPEND', 'sum'),
            transactions_mid_90=('BASKET_NUM', 'nunique')
        ).reset_index()
        
        # Lifetime aggregates
        lifetime_agg = full_data.groupby('HSHD_NUM').agg(
            spend_lifetime=('SPEND', 'sum'),
            transactions_lifetime=('BASKET_NUM', 'nunique'),
            first_purchase=('PURCHASE_', 'min'),
            last_purchase=('PURCHASE_', 'max')
        ).reset_index()
        
        # Merge all
        behavior = lifetime_agg.merge(recent_agg, on='HSHD_NUM', how='left')
        behavior = behavior.merge(mid_agg, on='HSHD_NUM', how='left')
        
        # Fill missing values
        behavior = behavior.fillna(0)
        
        # Recent engagement ratios
        behavior['spend_recent_ratio'] = behavior['spend_recent_30'] / (behavior['spend_mid_90'] + 1)
        behavior['transactions_recent_ratio'] = behavior['transactions_recent_30'] / (behavior['transactions_mid_90'] + 1)
        
        # Drop rates
        behavior['spend_drop_pct'] = (behavior['spend_mid_90'] - behavior['spend_recent_30']) / (behavior['spend_mid_90'] + 1)
        behavior['transactions_drop_pct'] = (behavior['transactions_mid_90'] - behavior['transactions_recent_30']) / (behavior['transactions_mid_90'] + 1)
        
        # Recent monetary per transaction
        behavior['avg_spend_per_transaction_recent'] = behavior['spend_recent_30'] / (behavior['transactions_recent_30'] + 1)
        
        # Churn = major drop (> 80% spend drop) OR no recent activity
        behavior['churn'] = np.where(
            (behavior['spend_drop_pct'] > 0.8) | (behavior['transactions_recent_30'] == 0),
            1, 0
        )
        
        # Feature Selection
        feature_cols = [
            'spend_recent_30', 'spend_mid_90', 'spend_recent_ratio', 'spend_drop_pct',
            'transactions_recent_30', 'transactions_mid_90', 'transactions_recent_ratio',
            'transactions_drop_pct', 'avg_spend_per_transaction_recent'
        ]
        
        X = behavior[feature_cols].values
        y = behavior['churn'].values
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        # Standardize Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Load model or use random forest
        try:
            model = joblib.load(os.path.join(MODELS_DIR, 'churn_model.pkl'))
        except:
            # Train model if not found
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # Choose a test instance to explain
        sample_idx = min(instance_idx, len(X_test_scaled)-1)
        
        # Generate explanation
        if explanation_type == 'lime':
            explainer = LimeTabularExplainer(
                training_data=X_train_scaled,
                feature_names=feature_cols,
                class_names=['active', 'churn'],
                mode='classification'
            )
            
            exp = explainer.explain_instance(
                X_test_scaled[sample_idx],
                model.predict_proba,
                num_features=5
            )
            plt = exp.as_pyplot_figure()
            
            # Generate the plot
            # plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.tight_layout()
            
            # Save to buffer
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            # plt.close()
            
            return send_file(img_buf, mimetype='image/png')
            
        elif explanation_type == 'pdp':
            # Generate partial dependence plot for the specified feature
            feature_idx = min(feature_idx, len(feature_cols)-1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                model,
                X_train_scaled,
                features=[feature_idx],
                feature_names=feature_cols,
                ax=ax
            )
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            # plt.close()
            
            return send_file(img_buf, mimetype='image/png')
            
        else:
            return jsonify({'error': 'Invalid explanation type'}), 400
            
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': traceback_str}), 500

@app.route('/api/basket-recommendations', methods=['GET'])
def basket_recommendations():
    try:
        # Get parameters
        basket_items = request.args.get('items', '').split(',')
        
        if not basket_items or basket_items[0] == '':
            return jsonify({'error': 'No items provided'}), 400
            
        # Load models and data
        mlb = pickle.load(open(os.path.join(MODELS_DIR, 'mlb.pkl'), 'rb'))
        category_importances = pickle.load(open(os.path.join(MODELS_DIR, 'category_importances.pkl'), 'rb'))
        
        # Get all categories that have models
        available_categories = list(category_importances.keys())
        
        # Find which categories from the basket match our available models
        matching_items = [item for item in basket_items if item in available_categories]
        
        # Get recommendations based on other items frequently bought with these
        recommendations = []
        
        for item in matching_items:
            # Get top 3 associated items for this category
            if item in category_importances:
                item_associations = sorted(
                    category_importances[item].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                for associated_item, score in item_associations:
                    if associated_item not in matching_items and associated_item not in [r['item'] for r in recommendations]:
                        recommendations.append({
                            'item': associated_item,
                            'score': float(score),
                            'based_on': item
                        })
        
        # Sort by score and return top 5
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:5]
        
        return jsonify({
            'basket_items': basket_items,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/seasonal-trends', methods=['GET'])
def seasonal_trends():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query for seasonal patterns
        query = """
        SELECT 
            t.WEEK_NUM, 
            p.DEPARTMENT, 
            p.COMMODITY,
            AVG(t.SPEND) as AVG_SPEND,
            AVG(t.UNITS) as AVG_UNITS
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        GROUP BY t.WEEK_NUM, p.DEPARTMENT, p.COMMODITY
        ORDER BY t.WEEK_NUM
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/brand-preferences', methods=['GET'])
def brand_preferences():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        SELECT 
            p.DEPARTMENT, 
            p.COMMODITY,
            SUM(t.SPEND) as TOTAL_SPEND,
            COUNT(DISTINCT h.HSHD_NUM) as UNIQUE_CUSTOMERS
        FROM transactions t
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        JOIN household h ON t.HSHD_NUM = h.HSHD_NUM
        GROUP BY p.DEPARTMENT, p.COMMODITY
        ORDER BY TOTAL_SPEND DESC
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()
# ========================
# Interactive Search
# ========================
@app.route('/search', methods=['GET'])
def search():
    hshd_num = request.args.get('hshd_num')
    if not hshd_num:
        return jsonify({'error': 'HSHD_NUM parameter required'}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        SELECT h.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, 
               p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS,
               t.STORE_R, t.WEEK_NUM, t.YEAR,
               h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE,
               h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
        FROM transactions t
        JOIN household h ON t.HSHD_NUM = h.HSHD_NUM
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE h.HSHD_NUM = ?
        ORDER BY h.HSHD_NUM, t.BASKET_NUM, t.DATE, 
                 t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY
        """
        
        cursor.execute(query, hshd_num)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(results)
    
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    global LOGGEDIN
    LOGGEDIN = True
    try:
        return jsonify({'message': 'Log in  successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================
# HTML Templates
# ========================
@app.route('/')
def index():
    if(LOGGEDIN):
        return render_template('index.html')
    else:
        return render_template('login.html')
if __name__ == '__main__':
    global LOGGEDIN
    LOGGEDIN = False
    app.run(debug=True)