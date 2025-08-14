#!/usr/bin/env python3
"""
Test script for Iris-Pupil Ratio Analyzer
This script verifies that all dependencies are properly installed and basic functionality works.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'cv2',
        'numpy',
        'mediapipe',
        'PIL',
        'reportlab'
    ]

    print("Testing package imports...")
    failed_imports = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n✗ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality without running the full app"""
    try:
        import cv2 as cv
        import numpy as np
        import mediapipe as mp

        # Test MediaPipe initialization
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        print("✓ MediaPipe Face Mesh initialized successfully")

        # Test OpenCV
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
        print("✓ OpenCV basic functionality working")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Iris-Pupil Ratio Analyzer - Test Suite")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n✓ All tests passed! The application should work correctly.")
            print("\nTo run the application:")
            print("streamlit run app.py")
        else:
            print("\n✗ Basic functionality tests failed.")
            sys.exit(1)
    else:
        print("\n✗ Import tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()