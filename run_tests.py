"""run_tests.py - Standalone test runner script for Smart Library Research Agent.

Runs the automated test suite and reports formatted results to console.
Can be executed via:
    python run_tests.py
"""

import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    tests_dir = project_root / "tests"

    print("=" * 70)
    print("📚 Running Smart Library Research Agent Automated Test Suite...")
    print("=" * 70)

    pytest_args = [
        str(tests_dir),
        "-v",
        "--tb=short",
        "-s",
    ]

    exit_code = pytest.main(pytest_args)
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print(f"\n❌ Tests exited with code: {exit_code}")

    sys.exit(exit_code)
