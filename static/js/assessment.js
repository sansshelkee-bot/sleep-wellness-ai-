document.addEventListener('DOMContentLoaded', function () {
    // Slider Sync
    setupSlider('sleep_duration', 'val_sleep_duration', ' hrs');
    setupSlider('quality_of_sleep', 'val_quality_of_sleep', ' / 10');
    setupSlider('stress_level', 'val_stress_level', ' / 10');
    setupSlider('physical_activity_level', 'val_activity', ' mins');
    setupSlider('daily_steps', 'val_steps', ' steps');

    const form = document.getElementById('assessmentForm');
    if (form) {
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            const btn = document.getElementById('submitAssessmentBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Processing AI Assessment...`;
            btn.disabled = true;

            const formData = new FormData(form);

            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success' && data.redirect_url) {
                    window.location.href = data.redirect_url;
                } else {
                    alert("Error: " + (data.message || "Failed to process assessment."));
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                }
            })
            .catch(err => {
                console.error("Submission error:", err);
                alert("An error occurred during submission. Please try again.");
                btn.innerHTML = originalText;
                btn.disabled = false;
            });
        });
    }
});

function setupSlider(sliderId, displayId, suffix) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (!slider || !display) return;

    slider.addEventListener('input', function () {
        let val = this.value;
        if (sliderId === 'daily_steps') {
            val = Number(val).toLocaleString();
        }
        display.textContent = val + suffix;
    });
}
