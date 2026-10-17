#!/usr/bin/env python3
"""
Verification script for Multi-Agent Analytics Platform.
Tests that all components are properly installed and configured.
"""
import sys
import os
from pathlib import Path

def test_project_structure():
    """Verify all expected directories and files exist."""
    print("🔍 Checking project structure...")

    required_paths = [
        "agents/__init__.py",
        "agents/base_agent.py",
        "agents/orchestrator_agent.py",
        "agents/data_collector_agent.py",
        "agents/quality_assurance_agent.py",
        "agents/data_processor_agent.py",
        "agents/analyzer_agent.py",
        "agents/insight_generator_agent.py",
        "agents/visualization_agent.py",
        "utils/__init__.py",
        "utils/config.py",
        "utils/logger.py",
        "utils/redis_client.py",
        "config.yaml",
        ".env.example",
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        "examples/complete_example.py"
    ]

    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    else:
        print("✅ All core files present")
        return True

def test_config_system():
    """Test that configuration loads correctly."""
    print("\n🔍 Testing configuration system...")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))

        from utils.config import Config

        config = Config()

        # Test getting values (will use defaults or .env.example)
        redis_host = config.get('orchestrator.redis.host', 'localhost')
        redis_port = config.get('orchestrator.redis.port', 6379)

        print(f"✅ Config system working - Redis: {redis_host}:{redis_port}")
        return True

    except Exception as e:
        print(f"❌ Config system error: {e}")
        return False

def test_basic_imports():
    """Test that basic modules can be imported."""
    print("\n🔍 Testing basic imports...")

    try:
        sys.path.insert(0, str(Path.cwd()))

        # Test utils
        from utils.config import Config
        from utils.logger import get_logger
        from utils.redis_client import RedisClient

        # Test that we can create instances
        config = Config()
        logger = get_logger(__name__)

        print("✅ Basic modules imported successfully")
        return True

    except Exception as e:
        print(f"❌ Basic import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_requirements():
    """Check if requirements.txt exists and has content."""
    print("\n🔍 Checking requirements.txt...")

    try:
        if Path("requirements.txt").exists():
            with open("requirements.txt", "r") as f:
                content = f.read().strip()
                if content:
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                    print(f"✅ requirements.txt found with {len(lines)} packages")
                    return True
                else:
                    print("⚠️ requirements.txt is empty")
                    return False
        else:
            print("❌ requirements.txt not found")
            return False
    except Exception as e:
        print(f"❌ Error reading requirements: {e}")
        return False

def test_example_file():
    """Check if example file exists."""
    print("\n🔍 Checking example file...")

    try:
        example_path = Path("examples/complete_example.py")
        if example_path.exists():
            print("✅ Example file found")
            return True
        else:
            print("❌ Example file not found")
            return False
    except Exception as e:
        print(f"❌ Error checking example: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("MULTI-AGENT PLATFORM VERIFICATION")
    print("=" * 60)

    tests = [
        test_project_structure,
        test_config_system,
        test_basic_imports,
        test_requirements,
        test_example_file
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Platform structure is ready!")
        print("\nTo test agent imports, you would need to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start Redis: docker run -d -p 6379:6379 redis")
        print("3. Then run: python -c \"import sys; sys.path.append('.'); from agents.orchestrator_agent import OrchestratorAgent; print('Import successful')\"")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your credentials")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start Redis: docker run -d -p 6379:6379 redis")
        print("4. Run example: python examples/complete_example.py")
        return True
    else:
        print("❌ Some tests failed - please check the output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)