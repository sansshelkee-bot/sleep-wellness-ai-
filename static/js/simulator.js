let simChartInstance = null;
let debounceTimer = null;

document.addEventListener('DOMContentLoaded', function () {
    setupSimSlider('sim_sleep_duration', 'val_sim_sleep_duration', ' hrs');
    setupSimSlider('sim_stress_level', 'val_sim_stress_level', ' / 10');
    setupSimSlider('sim_physical_activity_level', 'val_sim_activity', ' mins');
    setupSimSlider('sim_daily_steps', 'val_sim_steps', ' steps');
    setupSimSlider('sim_quality_of_sleep', 'val_sim_quality', ' / 10');

    // Trigger initial simulation
    runSimulation();

    // Auto-run simulation on input change (debounced 300ms)
    const inputs = document.querySelectorAll('.sim-input');
    inputs.forEach(input => {
        input.addEventListener('input', function () {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(runSimulation, 300);
        });
    });

    const runBtn = document.getElementById('runSimBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runSimulation);
    }
});

function setupSimSlider(sliderId, displayId, suffix) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (!slider || !display) return;

    slider.addEventListener('input', function () {
        let val = this.value;
        if (sliderId === 'sim_daily_steps') {
            val = Number(val).toLocaleString();
        }
        display.textContent = val + suffix;
    });
}

function runSimulation() {
    const payload = {
        age: document.getElementById('sim_age')?.value || 35,
        gender: document.getElementById('sim_gender')?.value || 'Male',
        occupation: document.getElementById('sim_occupation')?.value || 'Engineer',
        sleep_duration: parseFloat(document.getElementById('sim_sleep_duration')?.value || 7.5),
        quality_of_sleep: parseInt(document.getElementById('sim_quality_of_sleep')?.value || 8),
        physical_activity_level: parseFloat(document.getElementById('sim_physical_activity_level')?.value || 60),
        stress_level: parseInt(document.getElementById('sim_stress_level')?.value || 4),
        bmi_category: document.getElementById('sim_bmi_category')?.value || 'Normal',
        blood_pressure: document.getElementById('sim_blood_pressure')?.value || '120/80',
        heart_rate: parseInt(document.getElementById('sim_heart_rate')?.value || 70),
        daily_steps: parseInt(document.getElementById('sim_daily_steps')?.value || 8000)
    };

    const statusBadge = document.getElementById('sim_status_indicator');
    if (statusBadge) statusBadge.textContent = "Calculating scenario...";

    fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            updateSimResultsUI(data);
            if (statusBadge) statusBadge.textContent = "Simulation Synced";
        }
    })
    .catch(err => {
        console.error("Simulation error:", err);
        if (statusBadge) statusBadge.textContent = "Ready";
    });
}

function updateSimResultsUI(data) {
    // 1. Simulated Disorder Badge
    const disorderBadge = document.getElementById('sim_disorder_badge');
    if (disorderBadge) {
        disorderBadge.textContent = data.simulated_prediction;
        disorderBadge.className = 'badge px-3 py-2 fs-6 ' + 
            (data.simulated_prediction === 'None' ? 'bg-success' : 'bg-warning text-dark');
    }

    // 2. Simulated Score & Delta
    const scoreVal = document.getElementById('sim_score_val');
    if (scoreVal) {
        scoreVal.textContent = data.simulated_score.overall_score;
    }

    const diffBadge = document.getElementById('sim_score_diff');
    if (diffBadge) {
        const diff = data.score_diff;
        if (diff > 0) {
            diffBadge.textContent = `+${diff} pts improvement`;
            diffBadge.className = 'diff-badge-positive';
        } else if (diff < 0) {
            diffBadge.textContent = `${diff} pts decline`;
            diffBadge.className = 'diff-badge-negative';
        } else {
            diffBadge.textContent = `0.0 pts (no change)`;
            diffBadge.className = 'badge bg-secondary';
        }
    }

    // 3. Simulated Risk
    const riskBadge = document.getElementById('sim_risk_badge');
    if (riskBadge) {
        const rLevel = data.simulated_risk.risk_level;
        riskBadge.textContent = `${rLevel} Risk (${data.simulated_risk.risk_score}/100)`;
        riskBadge.className = 'badge px-3 py-2 ' +
            (rLevel === 'Low' ? 'bg-success' : (rLevel === 'Moderate' ? 'bg-warning text-dark' : 'bg-danger'));
    }

    // 4. Update Probability Comparison Chart
    updateSimChart(data.simulated_probabilities);

    // 5. Update Recommendations
    const recContainer = document.getElementById('sim_recommendations_container');
    if (recContainer && data.simulated_recommendations) {
        const recs = data.simulated_recommendations.immediate_actions || [];
        recContainer.innerHTML = recs.map(r => `
            <div class="p-2 mb-2 bg-light rounded border text-muted small">
                <i class="bi bi-arrow-right-circle text-primary me-1"></i> ${r.text}
            </div>
        `).join('');
    }
}

function updateSimChart(probs) {
    const ctx = document.getElementById('simProbChart');
    if (!ctx) return;

    const classes = ['None', 'Insomnia', 'Sleep Apnea'];
    const values = classes.map(c => Math.round((probs[c] || 0) * 100));

    if (simChartInstance) {
        simChartInstance.data.datasets[0].data = values;
        simChartInstance.update();
    } else {
        simChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['None (Healthy)', 'Insomnia', 'Sleep Apnea'],
                datasets: [{
                    label: 'Simulated Probability (%)',
                    data: values,
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: { callback: v => v + '%' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}
