import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI

import predictor.api_server as api_server

def test_app_defined():
    assert hasattr(api_server, 'app'), 'app attribute missing'
    assert isinstance(api_server.app, FastAPI)

