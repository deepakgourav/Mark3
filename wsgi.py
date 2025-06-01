import sys
import os

# Add your project folder to the system path
sys.path.insert(0, os.path.expanduser('~/my_flask_app'))

from app import app as application  # This should match your app's filename
