let examples = [];

// Charger les exemples au démarrage
fetch('/api/examples')
    .then(response => response.json())
    .then(data => {
        examples = data;
    })
    .catch(error => console.error('Erreur chargement exemples:', error));

// Charger un exemple
function loadExample(index) {
    if (examples[index]) {
        const data = examples[index].data;
        
        document.getElementById('title').value = data.title || '';
        document.getElementById('description').value = data.description || '';
        document.getElementById('price').value = data.price || '';
        document.getElementById('area_sqm').value = data.area_sqm || '';
        document.getElementById('location').value = data.location || 'Orchard';
        document.getElementById('account_age_days').value = data.account_age_days || '';
        document.getElementById('contact_verified').checked = data.contact_verified || false;
        
        // Calculer et afficher le prix au m²
        updatePricePerSqm();
    }
}

// Calculer prix au m² automatiquement
function updatePricePerSqm() {
    const price = parseFloat(document.getElementById('price').value) || 0;
    const area = parseFloat(document.getElementById('area_sqm').value) || 0;
    return area > 0 ? price / area : 0;
}

// Validation des données du formulaire
function validateFormData(data) {
    const errors = [];
    
    if (!data.title || data.title.trim().length === 0) {
        errors.push("Le titre est requis");
    }
    
    if (!data.description || data.description.trim().length === 0) {
        errors.push("La description est requise");
    }
    
    if (!data.price || data.price <= 0) {
        errors.push("Le prix doit être positif");
    }
    
    if (!data.area_sqm || data.area_sqm <= 0) {
        errors.push("La surface doit être positive");
    }
    
    if (!data.account_age_days || data.account_age_days < 0) {
        errors.push("L'âge du compte doit être positif");
    }
    
    return errors;
}

// Sécurisation de l'accès aux propriétés
function safeGet(obj, path, defaultValue = '') {
    return path.split('.').reduce((current, key) => {
        return (current && current[key] !== undefined) ? current[key] : defaultValue;
    }, obj);
}

// Gestion du formulaire avec sécurisation complète
document.getElementById('fraud-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const resultSection = document.getElementById('result-section');
    
    // Désactiver le bouton pendant l'analyse
    submitBtn.textContent = '⏳ Analyse en cours...';
    submitBtn.disabled = true;
    
    try {
        // Collecter les données du formulaire avec validation
        const formData = new FormData(e.target);
        const data = {
            title: (formData.get('title') || '').trim(),
            description: (formData.get('description') || '').trim(),
            price: parseFloat(formData.get('price')) || 0,
            area_sqm: parseFloat(formData.get('area_sqm')) || 0,
            price_per_sqm: updatePricePerSqm(),
            location: formData.get('location') || 'Orchard',
            account_age_days: parseInt(formData.get('account_age_days')) || 0,
            contact_verified: formData.get('contact_verified') === 'on'
        };
        
        // Validation côté client
        const validationErrors = validateFormData(data);
        if (validationErrors.length > 0) {
            displayError('Erreurs de validation: ' + validationErrors.join(', '));
            return;
        }
        
        console.log('Données envoyées:', data);
        
        // Envoyer la requête à l'API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        console.log('Réponse reçue:', result);
        
        if (response.ok) {
            displayResult(result);
        } else {
            const errorMessage = safeGet(result, 'error', 'Erreur inconnue');
            displayError(errorMessage);
        }
        
    } catch (error) {
        console.error('Erreur de connexion:', error);
        displayError('Erreur de connexion: ' + error.message);
    } finally {
        // Réactiver le bouton
        submitBtn.textContent = '🔍 Analyser l\'Annonce';
        submitBtn.disabled = false;
    }
});

function displayResult(result) {
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    
    // Sécurisation de tous les accès aux propriétés
    const isfraud = safeGet(result, 'is_fraud', false);
    const fraudProbability = safeGet(result, 'fraud_probability', 0);
    const confidenceLevel = safeGet(result, 'confidence_level', 'Non défini');
    const riskFactors = safeGet(result, 'risk_factors', []);
    const recommendation = safeGet(result, 'recommendation', 'NON_DÉFINI');
    
    // Validation des types
    const probability = typeof fraudProbability === 'number' ? fraudProbability : parseFloat(fraudProbability) || 0;
    const factors = Array.isArray(riskFactors) ? riskFactors : [];
    
    const fraudClass = isfraud ? 'fraud' : 'legitimate';
    const statusIcon = isfraud ? '⚠️' : '✅';
    const statusText = isfraud ? 'FRAUDE DÉTECTÉE' : 'ANNONCE LÉGITIME';
    
    // Sécurisation de la recommandation pour le CSS
    const safeRecommendation = typeof recommendation === 'string' ? recommendation.toLowerCase() : 'indefini';
    
    resultContent.innerHTML = `
        <div class="result-card ${fraudClass}">
            <div class="result-header">
                <h3>${statusIcon} ${statusText}</h3>
                <div class="probability">
                    Probabilité de fraude: <strong>${(probability * 100).toFixed(1)}%</strong>
                </div>
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <span class="label">Confiance:</span>
                    <span class="value">${confidenceLevel}</span>
                </div>
                
                <div class="detail-item">
                    <span class="label">Recommandation:</span>
                    <span class="value recommendation-${safeRecommendation}">${recommendation}</span>
                </div>
            </div>
            
            ${factors.length > 0 ? `
                <div class="risk-factors">
                    <h4>🚨 Facteurs de Risque Identifiés:</h4>
                    <ul>
                        ${factors.map(factor => `<li>${String(factor || 'Facteur non défini')}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${safeGet(result, 'error') ? `
                <div class="error-details">
                    <h4>⚠️ Détails de l'erreur:</h4>
                    <p>${safeGet(result, 'error', 'Erreur inconnue')}</p>
                </div>
            ` : ''}
        </div>
    `;
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

function displayError(error) {
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    
    // Sécurisation de l'affichage d'erreur
    const errorMessage = typeof error === 'string' ? error : 'Erreur inconnue';
    
    resultContent.innerHTML = `
        <div class="result-card error">
            <h3>❌ Erreur</h3>
            <p>${errorMessage}</p>
            <div class="error-actions">
                <button onclick="location.reload()" class="btn btn-secondary">Recharger la page</button>
            </div>
        </div>
    `;
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Gestionnaire d'erreurs global pour JavaScript
window.addEventListener('error', function(event) {
    console.error('Erreur JavaScript globale:', event.error);
    displayError('Erreur technique inattendue. Veuillez recharger la page.');
});

// Gestionnaire pour les promesses rejetées
window.addEventListener('unhandledrejection', function(event) {
    console.error('Promise rejetée:', event.reason);
    displayError('Erreur de connexion. Vérifiez votre connexion internet.');
});
