document.addEventListener('DOMContentLoaded', function () {
    fetchProgressCharts();
});

function fetchProgressCharts() {
    fetch('/api/progress')
        .then(response => response.json())
        .then(data => {
            if (!data.labels || data.labels.length === 0) {
                return;
            }
            renderScoreChart(data);
            renderSleepStressChart(data);
            renderActivityChart(data);
        })
        .catch(err => console.error("Error loading progress charts:", err));
}

function renderScoreChart(data) {
    const ctx = document.getElementById('scoreTrendChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Sleep Health Score',
                data: data.scores,
                borderColor: '#0284c7',
                backgroundColor: 'rgba(2, 132, 199, 0.08)',
                borderWidth: 2.5,
                fill: true,
                tension: 0.35,
                pointBackgroundColor: '#0284c7',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: { stepSize: 20 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function renderSleepStressChart(data) {
    const ctx = document.getElementById('sleepStressChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Sleep Duration (hrs)',
                    data: data.durations,
                    borderColor: '#10b981',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y'
                },
                {
                    label: 'Stress Level (1-10)',
                    data: data.stresses,
                    borderColor: '#ef4444',
                    borderDash: [5, 5],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 3,
                    max: 10,
                    title: { display: true, text: 'Sleep Duration (hrs)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 10,
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Stress Level' }
                }
            }
        }
    });
}

function renderActivityChart(data) {
    const ctx = document.getElementById('activityChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Daily Steps',
                    data: data.steps,
                    backgroundColor: 'rgba(99, 102, 241, 0.75)',
                    borderRadius: 6,
                    yAxisID: 'y'
                },
                {
                    type: 'line',
                    label: 'Active Mins',
                    data: data.activities,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Steps' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 120,
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Active Minutes' }
                }
            }
        }
    });
}
