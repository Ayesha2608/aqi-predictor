"""
Quick test script to verify MongoDB connection and environment setup.
Run this locally to ensure your credentials work before testing on GitHub Actions.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_mongodb_connection():
    """Test MongoDB connection with your credentials."""
    print("=" * 60)
    print("🔍 TESTING MONGODB CONNECTION")
    print("=" * 60)
    
    # Test 1: Check environment variables
    print("\n1️⃣ Checking Environment Variables...")
    
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db = os.getenv("MONGODB_DB", "aqi_predictor")
    aqi_city = os.getenv("AQI_CITY", "Karachi")
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not found in environment!")
        print("   Set it with: $env:MONGODB_URI='your_connection_string'")
        return False
    
    print(f"✅ MONGODB_URI: {mongodb_uri[:30]}...{mongodb_uri[-20:]}")
    print(f"✅ MONGODB_DB: {mongodb_db}")
    print(f"✅ AQI_CITY: {aqi_city}")
    
    # Test 2: Try to connect
    print("\n2️⃣ Testing MongoDB Connection...")
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        # Create client with proper settings
        client = MongoClient(mongodb_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        # Test database access
        db = client[mongodb_db]
        collections = db.list_collection_names()
        print(f"✅ Database '{mongodb_db}' accessible")
        print(f"   Collections found: {collections if collections else 'None (empty database)'}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Common fixes:")
        print("   - Check your MongoDB Atlas IP whitelist (allow 0.0.0.0/0 for GitHub Actions)")
        print("   - Verify username/password in connection string")
        print("   - Ensure cluster is running (not paused)")
        return False

def test_config_import():
    """Test if config settings can be imported."""
    print("\n3️⃣ Testing Config Import...")
    try:
        from config.settings import (
            MONGODB_DB,
            DEFAULT_CITY,
            OPENWEATHER_API_KEY
        )
        print("✅ Config loaded successfully")
        print(f"   City: {DEFAULT_CITY}")
        print(f"   DB: {MONGODB_DB}")
        print(f"   OpenWeather Key: {'Set ✅' if OPENWEATHER_API_KEY else 'Not set ⚠️'}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n🚀 AQI Predictor - Environment Verification")
    print("This script tests your local setup before deploying to GitHub.\n")
    
    results = []
    results.append(("Config Import", test_config_import()))
    results.append(("MongoDB Connection", test_mongodb_connection()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to deploy to GitHub Actions.")
        print("\n📝 Next steps:")
        print("   1. Add all secrets and variables to GitHub (see FINAL_STEPS.md)")
        print("   2. Go to Actions tab and manually trigger 'Continuous Integration'")
        print("   3. Check for green checkmarks ✅")
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before deploying.")
    
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
