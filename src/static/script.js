document.getElementById('transactionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form data
    const formData = {
        step: parseInt(document.getElementById('step').value),
        type: document.getElementById('type').value,
        amount: parseFloat(document.getElementById('amount').value),
        oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value),
        newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value),
        oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value),
        newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value)
    };

    try {
        // Show loading state
        const button = document.querySelector('button');
        button.disabled = true;
        button.textContent = 'Checking...';

        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'API request failed');
        }

        const result = await response.json();

        // Display result
        const resultDiv = document.getElementById('result');
        const fraudStatus = document.getElementById('fraudStatus');
        const fraudProbability = document.getElementById('fraudProbability');

        resultDiv.classList.remove('hidden');
        fraudStatus.textContent = result.is_fraud ? 'Fraudulent' : 'Legitimate';
        fraudStatus.className = result.is_fraud ? 'fraud' : 'no-fraud';
        fraudProbability.textContent = `${(result.fraud_probability * 100).toFixed(2)}%`;

    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    } finally {
        // Reset button state
        const button = document.querySelector('button');
        button.disabled = false;
        button.textContent = 'Check for Fraud';
    }
}); 