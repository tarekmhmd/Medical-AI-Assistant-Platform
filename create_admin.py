import sqlite3
import os
from werkzeug.security import generate_password_hash

db_path = r'D:\project 2\backend\medical_assistant.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if users table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
table_exists = cursor.fetchone()

if not table_exists:
    print("Creating users table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            age INTEGER,
            address TEXT,
            profile_image TEXT,
            google_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    print("Users table created")

# Check for admin user
cursor.execute('SELECT id, email, name FROM users WHERE email = ?', ('admin@admin.com',))
admin = cursor.fetchone()

if admin:
    print(f"Admin already exists: ID={admin[0]}, Email={admin[1]}, Name={admin[2]}")
else:
    # Create admin user
    hashed_password = generate_password_hash('admin123')
    cursor.execute('''
        INSERT INTO users (email, password, name, age, address)
        VALUES (?, ?, ?, ?, ?)
    ''', ('admin@admin.com', hashed_password, 'Administrator', 30, 'System'))
    conn.commit()
    print("Admin account created successfully!")

# Verify admin exists
cursor.execute('SELECT id, email, name FROM users WHERE email = ?', ('admin@admin.com',))
admin = cursor.fetchone()
print(f"\nAdmin account ready: Username=admin@admin.com, Password=admin123")
print(f"Database: {db_path}")

conn.close()
