import os
import csv
import io
import pyodbc
import numpy as np
import pandas as pd
import joblib
import pickle
import shap
import lime
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

# ==================================
# App and Configuration
# ==================================
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')  # Secret key for session management

# Models directory
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================================
# Database Configuration
# ==================================
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

# ==================================
# Upload Data
# ==================================
@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            transactions_file = request.files['transactions']
            households_file = request.files['households']
            products_file = request.files['products']

            conn = get_connection()
            cursor = conn.cursor()

            # Transactions
            cursor.execute("TRUNCATE TABLE transactions")
            transactions = list(csv.reader(transactions_file.read().decode('utf-8').splitlines()))
            cursor.executemany(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                transactions[1:]
            )

            # Households
            cursor.execute("TRUNCATE TABLE household")
            households = list(csv.reader(households_file.read().decode('utf-8').splitlines()))
            cursor.executemany(
                "INSERT INTO household VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                households[1:]
            )

            # Products
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

# ==================================
# Data Analysis APIs
# ==================================
@app.route('/analysis', methods=['GET'])
def analysis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('analysis.html')

@app.route('/api/demographic-engagement', methods=['GET'])
def demographic_engagement():
    try:
        conn = get_connection()
        cursor = conn.cursor()

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

# ==================================
# Interactive Search
# ==================================
@app.route('/search', methods=['GET'])
def search():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))  # Updated to correct login page route

    hshd_num = request.args.get('hshd_num')
    if not hshd_num:
        return jsonify({'error': 'HSHD_NUM parameter required'}), 400

    conn = None
    cursor = None
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
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================================
# Authentication: Login / Logout
# ==================================
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()  # <- This reads JSON body

        username = data.get('username')
        password = data.get('password')

        if username == 'admin' and password == 'password123':
            session['logged_in'] = True
            return jsonify({'message': 'Logged in successfully!'}), 200
        else:
            return jsonify({'message': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login_page'))


# ==================================
# Home Page
# ==================================
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template('index.html')

# ==================================
# Main
# ==================================
if __name__ == '__main__':
    app.run(debug=True)
