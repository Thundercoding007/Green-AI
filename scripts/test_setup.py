# File: scripts/test_setup.py
# Verify setup and test components before full training

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import numpy
        print("   ✅ numpy")
    except ImportError as e:
        print(f"   ❌ numpy: {e}")
        return False
    
    try:
        import pandas
        print("   ✅ pandas")
    except ImportError as e:
        print(f"   ❌ pandas: {e}")
        return False
    
    try:
        import sklearn
        print("   ✅ scikit-learn")
    except ImportError as e:
        print(f"   ❌ scikit-learn: {e}")
        return False
    
    try:
        import torch
        print(f"   ✅ torch (CUDA available: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"   ❌ torch: {e}")
        return False
    
    try:
        import transformers
        print("   ✅ transformers")
    except ImportError as e:
        print(f"   ❌ transformers: {e}")
        return False
    
    try:
        import codecarbon
        print("   ✅ codecarbon")
    except ImportError as e:
        print(f"   ❌ codecarbon: {e}")
        return False
    
    try:
        import fastapi
        print("   ✅ fastapi")
    except ImportError as e:
        print(f"   ❌ fastapi: {e}")
        return False
    
    try:
        import streamlit
        print("   ✅ streamlit")
    except ImportError as e:
        print(f"   ❌ streamlit: {e}")
        return False
    
    return True


def test_directory_structure():
    """Test directory structure"""
    print("\n🗂️  Testing directory structure...")
    
    from src.config import Config
    
    required_dirs = [
        Config.DATA_DIR,
        Config.RAW_DATA_DIR,
        Config.PROCESSED_DATA_DIR,
        Config.MODELS_DIR,
        Config.GREEN_MODEL_PATH,
        Config.MEDIUM_MODEL_PATH,
        Config.HEAVY_MODEL_PATH,
        Config.LOGS_DIR,
        Config.EMISSIONS_DIR
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"   ✅ {dir_path.relative_to(Config.PROJECT_ROOT)}")
        else:
            print(f"   ❌ {dir_path.relative_to(Config.PROJECT_ROOT)} (missing)")
            all_exist = False
    
    return all_exist


def test_database():
    """Test database setup"""
    print("\n💾 Testing database...")
    
    try:
        from src.database import init_db, SessionLocal, InferenceLog
        
        # Initialize database
        init_db()
        print("   ✅ Database initialized")
        
        # Test connection
        db = SessionLocal()
        db.query(InferenceLog).first()
        db.close()
        print("   ✅ Database connection working")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing utilities"""
    print("\n🧹 Testing preprocessing...")
    
    try:
        from src.utils.preprocessing import EmailPreprocessor
        
        preprocessor = EmailPreprocessor()
        
        sample_email = """
        From: john.doe@example.com
        Subject: Test Meeting
        
        Hi team, let's meet tomorrow at 2pm.
        Call me at +1-234-567-8900 or visit https://example.com
        """
        
        cleaned = preprocessor.process_email(sample_email, anonymize=True)
        
        # Check if PII was removed
        assert '[EMAIL]' in cleaned or 'example.com' not in cleaned
        assert '[PHONE]' in cleaned or '234-567' not in cleaned
        assert '[URL]' in cleaned or 'https://' not in cleaned
        
        print("   ✅ Email preprocessing working")
        print(f"   📄 Sample output: {cleaned[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Preprocessing error: {e}")
        return False


def test_dataset_preparation():
    """Test dataset preparation"""
    print("\n📊 Testing dataset preparation...")
    
    try:
        from scripts.prepare_dataset import create_synthetic_email_dataset
        import pandas as pd
        
        # Create small synthetic dataset
        df = create_synthetic_email_dataset(n_samples=100)
        
        print(f"   ✅ Created {len(df)} synthetic emails")
        print(f"   📋 Classes: {df['label'].unique().tolist()}")
        
        # Test preprocessing on it
        from src.utils.preprocessing import EmailPreprocessor
        preprocessor = EmailPreprocessor()
        
        df['processed'] = df['text'].apply(
            lambda x: preprocessor.process_email(x, anonymize=True)
        )
        
        print(f"   ✅ Preprocessing works on dataset")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Dataset preparation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_green_model():
    """Test green model training on tiny dataset"""
    print("\n🌱 Testing Green Model (quick test)...")
    
    try:
        from src.models.green_model import GreenModel
        import pandas as pd
        
        # Create tiny dataset
        data = {
            'text': [
                'Meeting tomorrow at 10am',
                'Sale! 50% off everything',
                'Your ticket has been received',
                'Coffee this weekend?'
            ] * 10,  # 40 samples
            'label': ['work', 'promotions', 'support', 'personal'] * 10
        }
        df = pd.DataFrame(data)
        
        # Train model
        model = GreenModel(max_features=100)
        metrics = model.train(
            X_train=df['text'][:30],
            y_train=df['label'][:30],
            X_val=df['text'][30:],
            y_val=df['label'][30:],
            calibrate=False  # Skip calibration for speed
        )
        
        # Test prediction
        result = model.predict_single("Meeting next week")
        
        print(f"   ✅ Model trained successfully")
        print(f"   🎯 Test prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        print(f"   ⏱️  Inference time: {result['inference_time_ms']:.2f}ms")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Green model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_availability():
    """Check if dataset is available"""
    print("\n📁 Checking dataset availability...")
    
    from src.config import Config
    
    # Check for processed data
    train_file = Config.PROCESSED_DATA_DIR / "train.csv"
    val_file = Config.PROCESSED_DATA_DIR / "val.csv"
    test_file = Config.PROCESSED_DATA_DIR / "test.csv"
    
    if train_file.exists() and val_file.exists() and test_file.exists():
        print("   ✅ Processed dataset found!")
        
        import pandas as pd
        train_df = pd.read_csv(train_file)
        print(f"   📊 Train samples: {len(train_df)}")
        print(f"   📋 Classes: {train_df['label'].unique().tolist()}")
        
        return True
    else:
        print("   ⚠️  No processed dataset found")
        print("   💡 Run: python scripts/prepare_dataset.py --synthetic")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("  🌿 GreenAI Email Classifier - Setup Verification")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['directories'] = test_directory_structure()
    results['database'] = test_database()
    results['preprocessing'] = test_preprocessing()
    results['dataset_prep'] = test_dataset_preparation()
    results['green_model'] = test_green_model()
    results['data_available'] = check_data_availability()
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 Test Summary")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Prepare dataset:")
        print("     python scripts/prepare_dataset.py --synthetic")
        print("  2. Train all models:")
        print("     python scripts/train_all_models.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing directories: Run setup.sh")
        print("  - Database issues: Delete greenai.db and retry")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)