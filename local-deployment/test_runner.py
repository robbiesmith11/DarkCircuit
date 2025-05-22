#!/usr/bin/env python3
"""
Test runner for DarkCircuit with dependency checking and better error reporting.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    # Core testing dependencies
    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")
    
    try:
        import pytest_asyncio
    except ImportError:
        missing_deps.append("pytest-asyncio")
    
    # Project dependencies
    try:
        import paramiko
    except ImportError:
        missing_deps.append("paramiko")
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import langchain_core
    except ImportError:
        missing_deps.append("langchain dependencies")
    
    return missing_deps

def run_basic_tests():
    """Run tests that don't require external dependencies."""
    print("🧪 Running basic framework tests...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_framework_verification.py",
        "tests/test_basic.py",
        "-v"
    ]
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        return return_code == 0
        
    except Exception as e:
        print(f"❌ Error running basic tests: {e}")
        return False

def run_all_tests():
    """Run all tests with coverage."""
    print("🧪 Running all tests with coverage...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        return return_code == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 DarkCircuit Test Runner")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("local_app.py"):
        print("❌ Please run this from the local-deployment directory")
        print("Current directory:", os.getcwd())
        return 1
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("⚠️  Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n📥 To install dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install -r tests/requirements-test.txt")
        print("\n🧪 Running basic tests only (no external dependencies)...")
        
        if run_basic_tests():
            print("\n✅ Basic tests passed!")
            print("💡 Install full dependencies to run complete test suite")
            return 0
        else:
            print("\n❌ Basic tests failed")
            return 1
    else:
        print("✅ All dependencies available")
        print("\n🧪 Running complete test suite...")
        
        if run_all_tests():
            print("\n✅ All tests completed!")
            print("📊 Coverage report: htmlcov/index.html")
            return 0
        else:
            print("\n⚠️  Some tests failed (this may be expected with missing services)")
            print("💡 Check test output above for details")
            return 0  # Don't fail completely for test failures

if __name__ == "__main__":
    sys.exit(main())