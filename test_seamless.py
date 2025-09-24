"""
Test script for the seamless interface
This tests the preloader and interface components
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_preloader():
    """Test the model preloader"""
    print("🧪 Testing Model Preloader...")
    
    try:
        from utils.preloader import ModelPreloader
        
        preloader = ModelPreloader()
        
        def progress_callback(model_name, progress, status):
            print(f"  {model_name}: {progress*100:.1f}% - {status}")
        
        preloader.set_progress_callback(progress_callback)
        
        print("📋 Available models:", preloader.available_models)
        print("🚀 Starting preload test (this will take a while)...")
        
        # Test with max_workers=1 for testing
        success = preloader.preload_all_models(max_workers=1)
        
        print(f"✅ Preloading {'successful' if success else 'failed'}")
        print(f"📦 Loaded models: {list(preloader.loaded_models.keys())}")
        
        # Test model data retrieval
        for model in preloader.available_models:
            is_loaded = preloader.is_model_loaded(model)
            print(f"  {model}: {'✅ Loaded' if is_loaded else '❌ Not loaded'}")
        
        # Cleanup
        preloader.cleanup_all()
        print("🧹 Cleanup completed")
        
        return success
        
    except Exception as e:
        print(f"❌ Preloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_components():
    """Test interface components without full launch"""
    print("\n🧪 Testing Interface Components...")
    
    try:
        from utils.seamless_interface import SeamlessVbotInterface
        from utils.welcome_screen import AvatarRecommender
        
        # Test avatar recommender
        recommender = AvatarRecommender()
        avatars = recommender.get_all_avatars()
        print(f"📋 Available avatars: {avatars}")
        
        # Test recommendations
        test_prefs = {"gender": "female", "personality": ["energetic", "curious"]}
        recommendation = recommender.recommend_avatar(test_prefs)
        print(f"🎯 Test recommendation: {recommendation}")
        
        print("✅ Interface components test passed")
        return True
        
    except Exception as e:
        print(f"❌ Interface components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Seamless Interface Test Suite")
    print("=" * 50)
    
    # Test interface components (quick)
    components_ok = test_interface_components()
    
    if not components_ok:
        print("❌ Basic components failed, skipping preloader test")
        return
    
    # Ask user if they want to run the full preloader test
    print("\n" + "=" * 50)
    print("⚠️  FULL PRELOADER TEST")
    print("This will load all 5 avatar models and may take 5-10 minutes.")
    print("It will use significant GPU memory and processing power.")
    
    response = input("Do you want to run the full preloader test? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        preloader_ok = test_preloader()
        
        if preloader_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("The seamless interface should work correctly.")
            print("\nTo launch the full application, run:")
            print("  python VbotSeamless.py")
        else:
            print("\n❌ Preloader test failed")
    else:
        print("\n⏭️  Skipped full preloader test")
        print("✅ Basic components are working")
        print("\nTo test the full seamless interface, run:")
        print("  python VbotSeamless.py")

if __name__ == "__main__":
    main()
