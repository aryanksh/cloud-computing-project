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