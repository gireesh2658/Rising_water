/**
 * =========================================================================
 * RISING WATERS — Frontend JavaScript
 * =========================================================================
 * Handles rain animation, form validation, progress bar animation,
 * and interactive micro-animations for the flood prediction web app.
 * =========================================================================
 */

// ── Rain Animation ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    createRainEffect();
    animateProgressBar();
    setupFormValidation();
});

/**
 * Creates animated raindrop elements in the rain container.
 * Generates between 40-80 raindrops with randomized positions,
 * sizes, and animation durations for a natural rain effect.
 */
function createRainEffect() {
    const container = document.getElementById('rainContainer');
    if (!container) return;

    const isHeavy = container.classList.contains('heavy-rain');
    const dropCount = isHeavy ? 80 : 40;

    for (let i = 0; i < dropCount; i++) {
        const drop = document.createElement('div');
        drop.classList.add('raindrop');
        drop.style.left = Math.random() * 100 + '%';
        drop.style.animationDuration = (Math.random() * 1.5 + 0.8) + 's';
        drop.style.animationDelay = Math.random() * 3 + 's';
        drop.style.opacity = Math.random() * 0.5 + 0.2;

        if (isHeavy) {
            drop.style.height = (Math.random() * 10 + 15) + 'px';
            drop.style.width = (Math.random() * 1 + 2) + 'px';
        } else {
            drop.style.height = (Math.random() * 10 + 10) + 'px';
            drop.style.width = (Math.random() * 1 + 1) + 'px';
        }

        container.appendChild(drop);
    }
}

/**
 * Animates the progress bar on the home page visual card.
 * Simulates a flood risk analysis progressing to completion.
 */
function animateProgressBar() {
    const progressFill = document.getElementById('progressFill');
    const barResult = document.getElementById('barResult');

    if (!progressFill || !barResult) return;

    // Animate progress bar after a short delay
    setTimeout(() => {
        progressFill.style.width = '72%';
    }, 800);

    // Update result text
    setTimeout(() => {
        barResult.textContent = 'Analysis Complete — Low Risk';
        barResult.style.color = '#43e97b';
    }, 2800);
}

/**
 * Sets up form validation and submission handling for the
 * prediction form. Adds visual feedback during submission.
 */
function setupFormValidation() {
    const form = document.getElementById('predictForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnLoader = document.getElementById('btnLoader');

    if (!form || !submitBtn) return;

    form.addEventListener('submit', function (e) {
        // Validate all fields have values
        const inputs = form.querySelectorAll('input[required]');
        let isValid = true;

        inputs.forEach(input => {
            if (!input.value || input.value.trim() === '') {
                isValid = false;
                input.style.borderColor = '#f5576c';
                input.style.boxShadow = '0 0 0 3px rgba(245, 87, 108, 0.15)';
            } else {
                input.style.borderColor = '';
                input.style.boxShadow = '';
            }
        });

        if (!isValid) {
            e.preventDefault();
            return;
        }

        // Show loading state
        if (btnLoader) {
            btnLoader.style.display = 'inline-block';
        }
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.7';
    });

    // Clear error styling on input
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('input', function () {
            this.style.borderColor = '';
            this.style.boxShadow = '';
        });

        // Add focus animation
        input.addEventListener('focus', function () {
            this.parentElement.style.transform = 'translateY(-2px)';
            this.parentElement.style.transition = '0.2s ease';
        });

        input.addEventListener('blur', function () {
            this.parentElement.style.transform = 'translateY(0)';
        });
    });
}

/**
 * Smooth scroll to element by ID.
 * Used for the "Learn More" button on the home page.
 * @param {string} elementId - The target element ID
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}
