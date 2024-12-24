async function loadCSV() {
    try {
        const response = await fetch('../data/raw_data.csv');
        const data = await response.text();

        const rows = data.split('\n').map(row => row.split(','));

        const countries = new Set();
        const crops = new Set();

        const headers = rows[0];
        const countryIndex = headers.indexOf("Area");
        const cropIndex = headers.indexOf("Item");

        for (let i = 1; i < rows.length; i++) {
            const row = rows[i];
            const country = row[countryIndex];
            const crop = row[cropIndex];

            if (country) countries.add(country);
            if (crop) crops.add(crop);
        }

        populateSelect('country', Array.from(countries));
        populateSelect('crop', Array.from(crops));
    } catch (error) {
        console.error('Error al cargar o procesar el CSV:', error);
    }
}

function populateSelect(id, data) {
    const select = document.getElementById(id);
    select.innerHTML = '<option value="">Seleccionar...</option>';
    data.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    });
}

window.onload = loadCSV;

document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("predictionForm");
    const predictBtn = document.getElementById("predictBtn");

    const resultElement = document.getElementById("result");
    const predictionResultElement = document.getElementById("predictionResult");

    form.addEventListener("submit", function(event) {
        event.preventDefault();

        const country = document.getElementById("country").value;
        const crop = document.getElementById("crop").value;
        const year = parseInt(document.getElementById("year").value);
        const averageRainfall = parseFloat(document.getElementById("averageRainfall").value);
        const pesticides = parseFloat(document.getElementById("pesticides").value);
        const avgTemp = parseFloat(document.getElementById("avgTemp").value);

        if (country === "" || crop === "" || isNaN(year) || isNaN(averageRainfall) || isNaN(pesticides) || isNaN(avgTemp)) {
            alert("Por favor, completa todos los campos.");
            return;
        }

        const inputData = {
            country: country,
            crop: crop,
            year: year,  
            averageRainfall: averageRainfall,
            pesticides: pesticides,
            avgTemp: avgTemp
        };

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
        })
        .then(response => response.json())
        .then(data => {
            const prediction = data.prediction;
            resultElement.textContent = `El rendimiento estimado para el cultivo de ${crop} en ${country} en el año ${year} es: ${prediction.toFixed(2)} kilogramos por hectárea.`;
            predictionResultElement.classList.remove("hidden");
        })
        .catch(error => {
            console.error('Error:', error);
            alert("Hubo un error al hacer la predicción.");
        });
    });
});
