// Global configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Add animated particles to body
function createParticles() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles';
    document.body.appendChild(particlesContainer);
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (15 + Math.random() * 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Add entrance animations
function addEntranceAnimations() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.1) + 's';
    });
    
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.15) + 's';
    });
}

// Initialize particles and animations on page load
if (document.body.classList.contains('login-page') || document.querySelector('.dashboard')) {
    document.addEventListener('DOMContentLoaded', () => {
        createParticles();
        addEntranceAnimations();
    });
}

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('token');
    const currentPage = window.location.pathname.split('/').pop();
    
    if (!token && currentPage !== 'index.html' && currentPage !== '') {
        window.location.href = 'index.html';
        return false;
    }
    
    if (token && (currentPage === 'index.html' || currentPage === '')) {
        window.location.href = 'dashboard.html';
        return false;
    }
    
    return true;
}

// Get auth headers
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    return {
        'Authorization': `Bearer ${token}`
    };
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `message ${type}`;
    notification.textContent = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.display = 'block';
    notification.style.minWidth = '300px';
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// File upload handler
function handleFileUpload(inputElement, callback) {
    inputElement.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            callback(file);
        }
    });
}

// Drag and drop handler
function setupDragAndDrop(dropZone, callback) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            callback(file);
        }
    });
}

// Show loading spinner
function showLoading(container) {
    container.innerHTML = '<div class="spinner"></div>';
}

// Hide loading spinner
function hideLoading(container) {
    const spinner = container.querySelector('.spinner');
    if (spinner) {
        spinner.remove();
    }
}

// API call wrapper
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers: {
                ...getAuthHeaders(),
                ...options.headers
            }
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Upload file to API
async function uploadFile(endpoint, file, fileFieldName = 'image') {
    const formData = new FormData();
    formData.append(fileFieldName, file);
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        return data;
    } catch (error) {
        console.error('Upload Error:', error);
        throw error;
    }
}

// Display analysis result
function displayResult(container, result) {
    if (result.error) {
        container.innerHTML = `
            <div class="message error">
                <strong>Error:</strong> ${result.error}
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="result-box">
            <div class="result-item">
                <div class="result-label">Diagnosis</div>
                <div class="result-value">${result.diagnosis}</div>
            </div>
            
            ${result.confidence ? `
            <div class="result-item">
                <div class="result-label">Confidence</div>
                <div class="result-value">${result.confidence}%</div>
            </div>
            ` : ''}
            
            <div class="result-item">
                <div class="result-label">Severity</div>
                <span class="severity-badge severity-${result.severity}">${result.severity.toUpperCase()}</span>
            </div>
            
            <div class="result-item">
                <div class="result-label">Treatment Recommendations</div>
                <div class="result-value">${result.treatment}</div>
            </div>
            
            ${result.recommendations && result.recommendations.length > 0 ? `
            <div class="result-item">
                <div class="result-label">Additional Recommendations</div>
                <ul style="margin-top: 10px; padding-left: 20px;">
                    ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        </div>
    `;
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Set active nav link
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPage) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
});


// Icon mapping for consistent usage across the app
const iconMap = {
    dashboard: 'fas fa-chart-line',
    profile: 'fas fa-user-circle',
    skin: 'fas fa-microscope',
    lab: 'fas fa-flask',
    chatbot: 'fas fa-robot',
    sound: 'fas fa-microphone-alt',
    records: 'fas fa-file-medical',
    about: 'fas fa-info-circle',
    contact: 'fas fa-envelope',
    logout: 'fas fa-sign-out-alt',
    upload: 'fas fa-cloud-upload-alt',
    analyze: 'fas fa-search-plus',
    send: 'fas fa-paper-plane',
    loading: 'fas fa-spinner fa-spin'
};

// Update all emoji icons to Font Awesome on page load
function updateIconsToFontAwesome() {
    // Update sidebar icons
    const navLinks = document.querySelectorAll('.nav-link span');
    const iconMapping = {
        '📊': iconMap.dashboard,
        '👤': iconMap.profile,
        '🔬': iconMap.skin,
        '🧪': iconMap.lab,
        '💬': iconMap.chatbot,
        '🎤': iconMap.sound,
        '📋': iconMap.records,
        'ℹ️': iconMap.about,
        '📧': iconMap.contact,
        '🚪': iconMap.logout
    };
    
    navLinks.forEach(span => {
        const emoji = span.textContent.trim();
        if (iconMapping[emoji]) {
            span.innerHTML = `<i class="${iconMapping[emoji]}"></i>`;
        }
    });
    
    // Update header icons
    const headers = document.querySelectorAll('.header h1, .card-header h2');
    headers.forEach(header => {
        const text = header.textContent;
        Object.keys(iconMapping).forEach(emoji => {
            if (text.includes(emoji)) {
                header.innerHTML = text.replace(emoji, `<i class="${iconMapping[emoji]}"></i>`);
            }
        });
    });
}

// Call icon update on DOM load
document.addEventListener('DOMContentLoaded', () => {
    updateIconsToFontAwesome();
});


// Add ripple effect to buttons
function createRipple(event) {
    const button = event.currentTarget;
    const ripple = document.createElement('span');
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    const radius = diameter / 2;
    
    ripple.style.width = ripple.style.height = `${diameter}px`;
    ripple.style.left = `${event.clientX - button.offsetLeft - radius}px`;
    ripple.style.top = `${event.clientY - button.offsetTop - radius}px`;
    ripple.classList.add('ripple-effect');
    
    const existingRipple = button.getElementsByClassName('ripple-effect')[0];
    if (existingRipple) {
        existingRipple.remove();
    }
    
    button.appendChild(ripple);
}

// Add ripple effect to all buttons
document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.btn, .btn-primary, .btn-send, .tab-btn');
    buttons.forEach(button => {
        button.addEventListener('click', createRipple);
    });
});

// Smooth scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Add scroll to top button
function addScrollToTopButton() {
    const scrollBtn = document.createElement('button');
    scrollBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollBtn.className = 'scroll-to-top';
    scrollBtn.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: var(--gradient-1);
        color: white;
        border: none;
        cursor: pointer;
        display: none;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        transition: all 0.3s;
    `;
    
    scrollBtn.addEventListener('click', scrollToTop);
    
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            scrollBtn.style.display = 'flex';
        } else {
            scrollBtn.style.display = 'none';
        }
    });
    
    document.body.appendChild(scrollBtn);
}

// Initialize scroll to top button
if (document.querySelector('.dashboard')) {
    document.addEventListener('DOMContentLoaded', addScrollToTopButton);
}

// Add loading animation to images
function addImageLoadingEffect() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.animation = 'fadeIn 0.5s ease-in';
        });
    });
}

document.addEventListener('DOMContentLoaded', addImageLoadingEffect);

// Enhanced notification with icons
function showNotificationEnhanced(message, type = 'info') {
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    
    const notification = document.createElement('div');
    notification.className = `message ${type} slide-in-up`;
    notification.innerHTML = `
        <i class="${icons[type]}" style="margin-right: 10px;"></i>
        ${message}
    `;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        display: flex;
        align-items: center;
        min-width: 300px;
        max-width: 500px;
        box-shadow: var(--shadow-lg);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideInDown 0.5s ease-out reverse';
        setTimeout(() => notification.remove(), 500);
    }, 5000);
}

// Add hover effect to cards
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});

// Add typing indicator for chatbot
function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message-bubble message-bot typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    return indicator;
}

// Animate numbers counting up
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        element.textContent = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Animate stat cards on dashboard
document.addEventListener('DOMContentLoaded', () => {
    const statCards = document.querySelectorAll('.stat-card h3');
    statCards.forEach(stat => {
        const finalValue = parseInt(stat.textContent) || 0;
        if (finalValue > 0) {
            animateValue(stat, 0, finalValue, 1500);
        }
    });
});

// Add parallax effect to background
function addParallaxEffect() {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('.particles');
        parallaxElements.forEach(element => {
            element.style.transform = `translateY(${scrolled * 0.5}px)`;
        });
    });
}

if (document.querySelector('.particles')) {
    addParallaxEffect();
}

// Preload images for better performance
function preloadImages() {
    const images = document.querySelectorAll('img[data-src]');
    images.forEach(img => {
        img.src = img.dataset.src;
        img.removeAttribute('data-src');
    });
}

document.addEventListener('DOMContentLoaded', preloadImages);
