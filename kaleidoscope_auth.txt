#!/usr/bin/env python3
"""
Kaleidoscope AI - Authentication Module
======================================
Handles user authentication for the web interface
"""

import os
import json
import uuid
import hashlib
import secrets
import logging
from typing import Dict, Optional, Any
from flask_login import UserMixin, LoginManager

logger = logging.getLogger(__name__)

# Initialize LoginManager for Flask
login_manager = LoginManager()

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, id: str, username: str, is_admin: bool = False):
        self.id = id
        self.username = username
        self.is_admin = is_admin
    
    def get_id(self):
        return self.id

class AuthManager:
    """Manages user authentication"""
    
    def __init__(self, config_dir: str):
        """
        Initialize the authentication manager
        
        Args:
            config_dir: Directory for storing user data
        """
        self.config_dir = config_dir
        self.users_file = os.path.join(config_dir, "users.json")
        self.users = {}
        self._load_users()
        
    def _load_users(self) -> None:
        """Load users from the users file"""
        os.makedirs(self.config_dir, exist_ok=True)
        
        if not os.path.exists(self.users_file):
            # Create default admin user
            default_password = secrets.token_urlsafe(12)
            self.create_user("admin", default_password, is_admin=True)
            logger.info(f"Created default admin user with password: {default_password}")
            logger.info("Please change this password immediately after logging in!")
            return
            
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users from configuration")
        except Exception as e:
            logger.error(f"Failed to load users: {str(e)}")
            self.users = {}
    
    def _save_users(self) -> None:
        """Save users to the users file"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {str(e)}")
    
    def create_user(self, username: str, password: str, is_admin: bool = False) -> bool:
        """
        Create a new user
        
        Args:
            username: Username
            password: Password
            is_admin: Whether the user is an admin
            
        Returns:
            True if user was created, False otherwise
        """
        if username in self.users:
            logger.warning(f"User {username} already exists")
            return False
            
        # Generate salt and hash password
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        # Create user
        self.users[username] = {
            "id": str(uuid.uuid4()),
            "username": username,
            "password_hash": password_hash,
            "salt": salt,
            "is_admin": is_admin
        }
        
        self._save_users()
        logger.info(f"Created user: {username}")
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication was successful, None otherwise
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
            
        user_data = self.users[username]
        salt = user_data["salt"]
        stored_hash = user_data["password_hash"]
        
        # Check password
        if self._hash_password(password, salt) != stored_hash:
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            return None
            
        # Create user object
        user = User(
            id=user_data["id"],
            username=user_data["username"],
            is_admin=user_data.get("is_admin", False)
        )
        
        logger.info(f"User {username} authenticated successfully")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        for username, user_data in self.users.items():
            if user_data["id"] == user_id:
                return User(
                    id=user_data["id"],
                    username=user_data["username"],
                    is_admin=user_data.get("is_admin", False)
                )
                
        return None
    
    def change_password(self, username: str, new_password: str) -> bool:
        """
        Change a user's password
        
        Args:
            username: Username
            new_password: New password
            
        Returns:
            True if password was changed, False otherwise
        """
        if username not in self.users:
            logger.warning(f"Cannot change password: User {username} not found")
            return False
            
        # Generate new salt and hash
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(new_password, salt)
        
        # Update user
        self.users[username]["password_hash"] = password_hash
        self.users[username]["salt"] = salt
        
        self._save_users()
        logger.info(f"Changed password for user: {username}")
        return True
    
    def _hash_password(self, password: str, salt: str) -> str:
        """
        Hash a password with salt
        
        Args:
            password: Password to hash
            salt: Salt to use
            
        Returns:
            Hashed password
        """
        combined = (password + salt).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()

# Initialize login manager
@login_manager.user_loader
def load_user(user_id):
    """Load a user from the database"""
    auth_manager = login_manager.auth_manager
    return auth_manager.get_user(user_id)

def init_auth(app, config_dir: str) -> AuthManager:
    """
    Initialize authentication
    
    Args:
        app: Flask application
        config_dir: Directory for storing user data
        
    Returns:
        AuthManager instance
    """
    auth_manager = AuthManager(config_dir)
    
    # Initialize login manager
    login_manager.auth_manager = auth_manager
    login_manager.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    
    return auth_manager
