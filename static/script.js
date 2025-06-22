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
        
        document.getElementById('title').value = data.title;
        document.getElementById('description').value = data.description;
        document.getElementById('price').value = data.price;
        document.getElementById('area_sqm').value = data.area_sqm;
        document.getElementById('location').value = data.location;
        document.getElementById('account_age_days').value = data.account_age_days;
        document.getElementById('contact_verified').checked = data.contact_verified;
        
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

// Gestion du formulaire
document.getElementById('fraud-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    
    // Désactiver le bouton pendant l'analyse
    submitBtn.textContent = '⏳ Analyse en cours...';
    submitBtn.disabled = true;
    
    try {
        // Collecter les données du formulaire
        const formData = new FormData(e.target);
        const data = {
            title: formData.get('title'),
            description: formData.get('description'),
            price: parseFloat(formData.get('price')),
            area_sqm: parseFloat(formData.get('area_sqm')),
            price_per_sqm: updatePricePerSqm(),
            location: formData.get('location'),
            account_age_days: parseInt(formData.get('account_age_days')),
            contact_verified: formData.get('contact_verified') === 'on'
        };
        
        // Envoyer la requête à l'API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResult(result);
        } else {
            displayError(result.error || 'Erreur inconnue');
        }
        
    } catch (error) {
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
    
    const fraudClass = result.is_fraud ? 'fraud' : 'legitimate';
    const statusIcon = result.is_fraud ? '⚠️' : '✅';
    const statusText = result.is_fraud ? 'FRAUDE DÉTECTÉE' : 'ANNONCE LÉGITIME';
    
    resultContent.innerHTML = `
        <div class="result-card ${fraudClass}">
            <div class="result-header">
                <h3>${statusIcon} ${statusText}</h3>
                <div class="probability">
                    Probabilité de fraude: <strong>${(result.fraud_probability * 100).toFixed(1)}%</strong>
                </div>
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <span class="label">Confiance:</span>
                    <span class="value">${result.confidence_level}</span>
                </div>
                
                <div class="detail-item">
                    <span class="label">Recommandation:</span>
                    <span class="value recommendation-${result.recommendation.toLowerCase()}">${result.recommendation || 'Non défini'}</span>
                </div>
            </div>
            
            ${result.risk_factors && result.risk_factors.length > 0 ? `
                <div class="risk-factors">
                    <h4>🚨 Facteurs de Risque Identifiés:</h4>
                    <ul>
                        ${result.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
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
    
    resultContent.innerHTML = `
        <div class="result-card error">
            <h3>❌ Erreur</h3>
            <p>${error}</p>
        </div>
    `;
    
    resultSection.style.display = 'block';
}
