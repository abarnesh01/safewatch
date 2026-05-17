import os
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
from database.db_manager import DatabaseManager
from loguru import logger

class AuthManager:
    """Handles authentication, session timeouts, and role-based access control."""

    def __init__(self, db_manager: DatabaseManager, secret_key: str, timeout_minutes: int = 60):
        self.db = db_manager
        self.secret_key = secret_key
        self.timeout_minutes = timeout_minutes
        self._ensure_admin_exists()

    def _ensure_admin_exists(self):
        """Creates default admin if no users exist in database."""
        # Check if users table exists
        self._create_users_table()
        
        cursor = self.db.execute("SELECT count(*) FROM users")
        count = cursor.fetchone()[0] if cursor else 0
        if count == 0:
            admin_user = "admin"
            admin_pass = "admin123"
            hashed_pw = self.hash_password(admin_pass)
            self.db.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (admin_user, hashed_pw, "Admin")
            )
            logger.info("Default admin user seeded into database.")

    def _create_users_table(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                last_login TEXT,
                failed_attempts INTEGER DEFAULT 0
            )
        """)

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def check_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def login(self, username, password):
        cursor = self.db.execute("SELECT id, password_hash, role, failed_attempts FROM users WHERE username = ?", (username,))
        user = cursor.fetchone() if cursor else None

        if not user:
            logger.warning(f"Failed login attempt for unknown user: {username}")
            return False, "Invalid username or password"
        
        user_id, pwd_hash, role, failed_attempts = user

        if failed_attempts >= 5:
            return False, "Account locked due to too many failed attempts"

        if self.check_password(password, pwd_hash):
            # Reset failed attempts
            self.db.execute("UPDATE users SET failed_attempts = 0, last_login = ? WHERE id = ?", (datetime.now().isoformat(), user_id))
            
            # Setup session
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.session_state['login_time'] = datetime.now()
            logger.info(f"User {username} logged in successfully as {role}.")
            return True, "Success"
        else:
            # Increment failed attempts
            self.db.execute("UPDATE users SET failed_attempts = failed_attempts + 1 WHERE id = ?", (user_id,))
            logger.warning(f"Failed login attempt for user: {username}")
            return False, "Invalid username or password"

    def logout(self):
        if 'authenticated' in st.session_state:
            username = st.session_state.get('username')
            logger.info(f"User {username} logged out.")
            for key in ['authenticated', 'username', 'role', 'login_time']:
                if key in st.session_state:
                    del st.session_state[key]

    def check_session(self) -> bool:
        """Validates if the current session is active and not timed out."""
        if not st.session_state.get('authenticated', False):
            return False
        
        login_time = st.session_state.get('login_time')
        if login_time:
            elapsed = datetime.now() - login_time
            if elapsed > timedelta(minutes=self.timeout_minutes):
                self.logout()
                st.warning("Session expired. Please log in again.")
                return False
        return True

    def require_role(self, allowed_roles: list) -> bool:
        """Checks if the logged-in user has one of the required roles."""
        if not self.check_session():
            return False
        user_role = st.session_state.get('role')
        return user_role in allowed_roles
