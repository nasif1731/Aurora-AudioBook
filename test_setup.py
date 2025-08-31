#!/usr/bin/env python3
"""
Test script to verify VoiceMyBook setup
Run this to check if all dependencies are properly installed
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_python_packages():
    """Test if all required Python packages can be imported"""
    packages = [
        'gradio',
        'fitz',  
        'dotenv',
        'pydub',
        'requests'
    ]
    
    print("TESTING: Python packages...")
    failed = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"  PASS: {package}")
        except ImportError as e:
            print(f"  FAIL: {package}: {e}")
            failed.append(package)
    
    return len(failed) == 0

def test_system_tools():
    """Test if system tools are available"""
    tools = [
        ('ffmpeg', 'ffmpeg -version'),
        ('tesseract', 'tesseract --version'),
        ('ocrmypdf', 'ocrmypdf --version')
    ]
    
    print("\nTESTING: System tools...")
    failed = []
    
    for tool_name, cmd in tools:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"  PASS: {tool_name}")
            else:
                print(f"  FAIL: {tool_name}: Command failed")
                failed.append(tool_name)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  FAIL: {tool_name}: {e}")
            failed.append(tool_name)
    
    return len(failed) == 0

def test_directory_structure():
    """Test if required directories can be created"""
    print("\nTESTING: Directory structure...")
    
    try:
        base = Path("./data")
        dirs = [
            base / "inputs",
            base / "outputs", 
            base / "outputs" / "audio_chunks",
            base / "outputs" / "meta"
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  PASS: {d}")
        
        return True
    except Exception as e:
        print(f"  FAIL: Directory creation failed: {e}")
        return False

def test_openvoice():
    """Test OpenVoice CLI availability"""
    print("\nTESTING: OpenVoice...")
    
    try:
        
        result = subprocess.run(
            'python -c "import openvoice_cli"', 
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  PASS: OpenVoice CLI (module)")
            return True
    except:
        pass
    
    try:
        
        result = subprocess.run(
            'openvoice-cli --help', 
            shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("  PASS: OpenVoice CLI (binary)")
            return True
    except:
        pass
    
    print("  WARNING: OpenVoice CLI not found (will need to install)")
    return False

def main():
    """Run all tests"""
    print("AudioBook Creator Setup Test")
    print("=" * 40)
    
    tests = [
        ("Python Packages", test_python_packages),
        ("System Tools", test_system_tools),
        ("Directory Structure", test_directory_structure),
        ("OpenVoice", test_openvoice)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  FAIL: {test_name}: Test failed with error: {e}")
            results.append((test_name, False))
    
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! AudioBook Creator is ready to use.")
        print("\nNext steps:")
        print("1. Create a .env file with your DEEPGRAM_API_KEY")
        print("2. Run: python app.py")
    else:
        print("\nWARNING: Some tests failed. Please fix the issues above before running AudioBook Creator.")
        print("\nCommon solutions:")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Install system tools (see README.md for instructions)")
        print("- Check if OpenVoice is properly installed")

if __name__ == "__main__":
    main()
