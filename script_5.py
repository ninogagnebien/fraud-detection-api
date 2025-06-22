# Corriger l'erreur et régénérer le rapport
from sklearn.metrics import classification_report

def corrected_model_report(trainer, results):
    """Rapport corrigé sans l'argument indent"""
    print("\n=== RAPPORT FINAL DES MODÈLES ===")
    
    best_auc = 0
    best_model = None
    
    for model_name, result in results.items():
        auc = result['auc']
        print(f"\n--- {model_name.upper()} ---")
        print(f"  AUC: {auc:.4f}")
        
        # Classification report sans indent
        y_pred_binary = (result['predictions'] > 0.5).astype(int)
        print("  Classification Report:")
        print(classification_report(result['test_labels'], y_pred_binary))
        
        if auc > best_auc:
            best_auc = auc
            best_model = model_name
    
    print(f"\n🎯 MEILLEUR MODÈLE: {best_model.upper()} (AUC: {best_auc:.4f})")
    return best_model, best_auc

# Générer le rapport corrigé
best_model, best_auc = corrected_model_report(trainer, supervised_results)

print(f"\n✅ Entraînement terminé avec succès!")
print(f"✅ Modèles sauvegardés dans le dossier models/")
print(f"✅ Meilleur modèle: {best_model} avec AUC = {best_auc:.4f}")