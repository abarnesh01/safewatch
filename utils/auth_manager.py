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
        self.users = {
            "admin": self.hash_password("admin123")
        }

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def check_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def login(self, username, password):
        if username not in self.users:
            logger.warning(f"Failed login attempt for unknown user: {username}")
            return False, "Invalid username or password"

        hashed_pw = self.users[username]
        
        if self.check_password(password, hashed_pw):
            # Setup session
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['role'] = "Admin"
            st.session_state['login_time'] = datetime.now()
            logger.info(f"User {username} logged in successfully as Admin.")
            return True, "Success"
        else:
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
