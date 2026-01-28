"""
Script de diagnostic GPU pour XGBoost et LightGBM
"""

print("\n" + "="*70)
print("🔍 DIAGNOSTIC GPU")
print("="*70 + "\n")

# 1. Vérifier CUDA
print("1️⃣  Vérification CUDA...")
import subprocess
try:
    result = subprocess.check_output('nvidia-smi -q', shell=True, text=True)
    cuda_version = [line for line in result.split('\n') if 'CUDA Version' in line]
    if cuda_version:
        print(f"   ✅ {cuda_version[0].strip()}")
except:
    print("   ❌ CUDA non détecté")

# 2. Vérifier XGBoost
print("\n2️⃣  Vérification XGBoost...")
import xgboost as xgb
print(f"   ✅ XGBoost version: {xgb.__version__}")

# Test si GPU fonctionne
try:
    from xgboost import XGBClassifier
    import numpy as np
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Test GPU
    model_gpu = XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=10,
        eval_metric='logloss'
    )
    model_gpu.fit(X, y)
    print("   ✅ XGBoost GPU: FONCTIONNE!")
except Exception as e:
    print(f"   ❌ XGBoost GPU échoue: {str(e)}")
    print("   💡 Essayez avec tree_method='hist' (CPU)")

# 3. Vérifier LightGBM
print("\n3️⃣  Vérification LightGBM...")
import lightgbm as lgb
print(f"   ✅ LightGBM version: {lgb.__version__}")

# Test si GPU fonctionne
try:
    from lightgbm import LGBMClassifier
    
    model_gpu = LGBMClassifier(
        device='gpu',
        n_estimators=10,
        verbose=-1
    )
    model_gpu.fit(X, y)
    print("   ✅ LightGBM GPU: FONCTIONNE!")
except Exception as e:
    print(f"   ❌ LightGBM GPU échoue: {str(e)}")
    print("   💡 LightGBM GPU n'est souvent pas compilé avec support NVIDIA")

# 4. Résumé
print("\n" + "="*70)
print("📊 RÉSUMÉ")
print("="*70)
print("""
Si XGBoost GPU ne marche pas:
  - Utilisez tree_method='hist' (CPU - rapide)
  - Ou réinstallez XGBoost: pip install xgboost-gpu

Si LightGBM GPU ne marche pas:
  - C'est normal, LightGBM GPU est rarement compilé
  - Utilisez LightGBM CPU (déjà rapide)
""")
