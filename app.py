import os
import csv
from flask import Flask, render_template, request, jsonify
import pyodbc

app = Flask(__name__)

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

# ========================
# HTML Templates
# ========================
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)