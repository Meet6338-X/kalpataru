/**
 * AgriOracle — Kalpataru Labs
 * Premium Agricultural AI Platform
 * JavaScript Application
 */

// API Configuration
const API_BASE = 'http://localhost:5000/api';

// DOM Elements
const menuBtn = document.getElementById('menuBtn');
const mobileMenu = document.getElementById('mobileMenu');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const clearChat = document.getElementById('clearChat');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const removeImage = document.getElementById('removeImage');
const moduleModal = document.getElementById('moduleModal');
const modalBody = document.getElementById('modalBody');
const closeModal = document.getElementById('closeModal');

// State
let selectedImage = null;
let conversationHistory = [];

// ============================================
// Navigation
// ============================================

menuBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('active');
    menuBtn.classList.toggle('active');
});

// Close mobile menu on link click
document.querySelectorAll('.mobile-link').forEach(link => {
    link.addEventListener('click', () => {
        mobileMenu.classList.remove('active');
        menuBtn.classList.remove('active');
    });
});

// Active nav link on scroll
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (scrollY >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ============================================
// Scroll Animations
// ============================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Observe module cards
document.querySelectorAll('.module-card').forEach((card, index) => {
    card.style.transitionDelay = `${index * 0.1}s`;
    observer.observe(card);
});

// ============================================
// Chat Functionality
// ============================================

// Send message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message && !selectedImage) return;

    // Add user message to UI
    addMessageToUI('user', message, selectedImage);
    
    // Clear input
    chatInput.value = '';
    const imageToSend = selectedImage;
    clearImagePreview();

    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';
    loadingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    scrollToBottom();

    try {
        let response;
        
        if (imageToSend) {
            // Send with image
            const formData = new FormData();
            formData.append('message', message || 'Analyze this image');
            formData.append('image', imageToSend);
            
            response = await fetch(`${API_BASE}/ai/chat/image`, {
                method: 'POST',
                body: formData
            });
        } else {
            // Text only
            response = await fetch(`${API_BASE}/ai/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
        }

        const data = await response.json();
        
        // Remove loading
        chatMessages.removeChild(loadingDiv);
        
        if (data.success) {
            addMessageToUI('bot', data.data.response);
        } else {
            addMessageToUI('bot', data.data?.fallback_response || 'Sorry, I encountered an error. Please try again.');
        }
    } catch (error) {
        chatMessages.removeChild(loadingDiv);
        addMessageToUI('bot', 'Sorry, I couldn\'t connect to the server. Please check your connection and try again.');
        console.error('Chat error:', error);
    }
}

// Add message to UI
function addMessageToUI(type, content, image = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = type === 'user' ? '👤' : '🤖';
    
    let imageHtml = '';
    if (image) {
        const imageUrl = URL.createObjectURL(image);
        imageHtml = `<img src="${imageUrl}" alt="Uploaded image">`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-text">${formatMessage(content)}</div>
            ${imageHtml}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Format message using marked.js for proper markdown rendering
function formatMessage(text) {
    if (!text) return '';
    
    // Configure marked options
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
        return marked.parse(text);
    }
    
    // Fallback to basic formatting if marked.js not loaded
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>')
        .replace(/- (.*?)(?=<br>|$)/g, '• $1');
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Clear chat
clearChat.addEventListener('click', async () => {
    chatMessages.innerHTML = `
        <div class="message bot">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">Welcome to **AgriOracle**! I'm your intelligent agricultural assistant. Ask me anything about:

- 🌱 **Crops** - Recommendations and best practices
- 🦠 **Diseases** - Detection and treatment
- 🧪 **Fertilizers** - NPK recommendations
- 🏔️ **Soil** - Classification and health
- 💧 **Irrigation** - Smart scheduling

You can also **upload an image** for analysis!</div>
            </div>
        </div>
    `;
    
    try {
        await fetch(`${API_BASE}/ai/clear`, { method: 'POST' });
    } catch (error) {
        console.error('Clear error:', error);
    }
});

// Quick prompts
document.querySelectorAll('.quick-prompt').forEach(btn => {
    btn.addEventListener('click', () => {
        chatInput.value = btn.dataset.prompt;
        sendMessage();
    });
});

// ============================================
// Image Upload
// ============================================

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.querySelector('img').src = e.target.result;
            imagePreview.classList.add('active');
        };
        reader.readAsDataURL(file);
    }
});

removeImage.addEventListener('click', clearImagePreview);

function clearImagePreview() {
    selectedImage = null;
    imagePreview.classList.remove('active');
    imagePreview.querySelector('img').src = '';
    imageInput.value = '';
}

// ============================================
// Module Cards & Modal
// ============================================

const moduleTemplates = {
    'ai-assistant': {
        title: 'AI ASSISTANT',
        icon: '🤖',
        content: `
            <p>Chat with our intelligent agricultural assistant. Ask about crops, diseases, fertilizers, and more!</p>
            <a href="#ai-assistant" class="btn btn-primary" onclick="closeModuleModal()">START CHATTING</a>
        `
    },
    'disease': {
        title: 'DISEASE DETECTION',
        icon: '🦠',
        content: `
            <form id="diseaseForm">
                <div class="form-group">
                    <label class="form-label">UPLOAD PLANT IMAGE</label>
                    <input type="file" class="form-input" id="diseaseImage" accept="image/*" required>
                </div>
                <button type="submit" class="btn btn-primary">DETECT DISEASE</button>
            </form>
            <div id="diseaseResult"></div>
        `
    },
    'crop': {
        title: 'CROP RECOMMENDATION',
        icon: '🌱',
        content: `
            <form id="cropForm">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">NITROGEN (N)</label>
                        <input type="number" class="form-input" id="cropN" value="50" min="0" max="200">
                    </div>
                    <div class="form-group">
                        <label class="form-label">PHOSPHORUS (P)</label>
                        <input type="number" class="form-input" id="cropP" value="50" min="0" max="200">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">POTASSIUM (K)</label>
                        <input type="number" class="form-input" id="cropK" value="50" min="0" max="200">
                    </div>
                    <div class="form-group">
                        <label class="form-label">pH</label>
                        <input type="number" class="form-input" id="cropPH" value="7" min="0" max="14" step="0.1">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">TEMPERATURE (°C)</label>
                        <input type="number" class="form-input" id="cropTemp" value="25">
                    </div>
                    <div class="form-group">
                        <label class="form-label">HUMIDITY (%)</label>
                        <input type="number" class="form-input" id="cropHumidity" value="60">
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">RAINFALL (mm)</label>
                    <input type="number" class="form-input" id="cropRainfall" value="1000">
                </div>
                <button type="submit" class="btn btn-primary">GET RECOMMENDATIONS</button>
            </form>
            <div id="cropResult"></div>
        `
    },
    'fertilizer': {
        title: 'FERTILIZER RECOMMENDATION',
        icon: '🧪',
        content: `
            <form id="fertilizerForm">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">CROP TYPE</label>
                        <select class="form-select" id="fertCrop">
                            <option value="Rice">Rice</option>
                            <option value="Wheat">Wheat</option>
                            <option value="Maize">Maize</option>
                            <option value="Cotton">Cotton</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">SOIL TYPE</label>
                        <select class="form-select" id="fertSoil">
                            <option value="Alluvial">Alluvial</option>
                            <option value="Black">Black</option>
                            <option value="Red">Red</option>
                            <option value="Sandy">Sandy</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">NITROGEN</label>
                        <input type="number" class="form-input" id="fertN" value="40">
                    </div>
                    <div class="form-group">
                        <label class="form-label">PHOSPHORUS</label>
                        <input type="number" class="form-input" id="fertP" value="20">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">POTASSIUM</label>
                        <input type="number" class="form-input" id="fertK" value="30">
                    </div>
                    <div class="form-group">
                        <label class="form-label">MOISTURE (%)</label>
                        <input type="number" class="form-input" id="fertMoisture" value="50">
                    </div>
                </div>
                <button type="submit" class="btn btn-primary">GET RECOMMENDATION</button>
            </form>
            <div id="fertilizerResult"></div>
        `
    },
    'weather': {
        title: 'WEATHER FORECAST',
        icon: '🌤️',
        content: `
            <form id="weatherForm">
                <div class="form-group">
                    <label class="form-label">LOCATION</label>
                    <input type="text" class="form-input" id="weatherLocation" value="Delhi" placeholder="Enter city name">
                </div>
                <button type="submit" class="btn btn-primary">GET WEATHER</button>
            </form>
            <div id="weatherResult"></div>
        `
    },
    'yield': {
        title: 'YIELD PREDICTION',
        icon: '📈',
        content: `
            <form id="yieldForm">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">CROP</label>
                        <select class="form-select" id="yieldCrop">
                            <option value="Rice">Rice</option>
                            <option value="Wheat">Wheat</option>
                            <option value="Maize">Maize</option>
                            <option value="Cotton">Cotton</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">AREA (HECTARES)</label>
                        <input type="number" class="form-input" id="yieldArea" value="1" step="0.1">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">REGION</label>
                        <select class="form-select" id="yieldRegion">
                            <option value="Punjab">Punjab</option>
                            <option value="Haryana">Haryana</option>
                            <option value="Uttar Pradesh">Uttar Pradesh</option>
                            <option value="Maharashtra">Maharashtra</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">SEASON</label>
                        <select class="form-select" id="yieldSeason">
                            <option value="Kharif">Kharif</option>
                            <option value="Rabi">Rabi</option>
                            <option value="Zaid">Zaid</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary">PREDICT YIELD</button>
            </form>
            <div id="yieldResult"></div>
        `
    },
    'soil': {
        title: 'SOIL CLASSIFICATION',
        icon: '🏔️',
        content: `
            <form id="soilForm">
                <div class="form-group">
                    <label class="form-label">UPLOAD SOIL IMAGE</label>
                    <input type="file" class="form-input" id="soilImage" accept="image/*" required>
                </div>
                <button type="submit" class="btn btn-primary">CLASSIFY SOIL</button>
            </form>
            <div id="soilResult"></div>
        `
    },
    'risk': {
        title: 'RISK ASSESSMENT',
        icon: '📊',
        content: `
            <form id="riskForm">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">CROP TYPE</label>
                        <select class="form-select" id="riskCrop">
                            <option value="Rice">Rice</option>
                            <option value="Wheat">Wheat</option>
                            <option value="Maize">Maize</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">COVERAGE (₹)</label>
                        <input type="number" class="form-input" id="riskCoverage" value="50000">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">WEATHER RISK (0-1)</label>
                        <input type="range" class="form-input" id="riskWeather" value="0.5" min="0" max="1" step="0.1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">CROP SUCCESS RATE (0-1)</label>
                        <input type="range" class="form-input" id="riskSuccess" value="0.7" min="0" max="1" step="0.1">
                    </div>
                </div>
                <button type="submit" class="btn btn-primary">CALCULATE RISK</button>
            </form>
            <div id="riskResult"></div>
        `
    },
    'irrigation': {
        title: 'IRRIGATION',
        icon: '💧',
        content: `
            <form id="irrigationForm">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">SOIL MOISTURE (%)</label>
                        <input type="number" class="form-input" id="irrMoisture" value="50">
                    </div>
                    <div class="form-group">
                        <label class="form-label">TEMPERATURE (°C)</label>
                        <input type="number" class="form-input" id="irrTemp" value="25">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">CROP TYPE</label>
                        <select class="form-select" id="irrCrop">
                            <option value="Rice">Rice</option>
                            <option value="Wheat">Wheat</option>
                            <option value="Maize">Maize</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">GROWTH STAGE</label>
                        <select class="form-select" id="irrStage">
                            <option value="Seedling">Seedling</option>
                            <option value="Vegetative">Vegetative</option>
                            <option value="Flowering">Flowering</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary">GET RECOMMENDATION</button>
            </form>
            <div id="irrigationResult"></div>
        `
    }
};

// Open module modal
document.querySelectorAll('.module-card').forEach(card => {
    card.addEventListener('click', () => {
        const moduleName = card.dataset.module;
        const template = moduleTemplates[moduleName];
        
        if (template) {
            modalBody.innerHTML = `
                <div class="modal-header">
                    <span class="modal-icon">${template.icon}</span>
                    <h2 class="modal-title">${template.title}</h2>
                </div>
                <div class="modal-content-inner">
                    ${template.content}
                </div>
            `;
            
            moduleModal.classList.add('active');
            initModuleForms();
        }
    });
});

// Close modal
closeModal.addEventListener('click', closeModuleModal);
moduleModal.addEventListener('click', (e) => {
    if (e.target === moduleModal) {
        closeModuleModal();
    }
});

function closeModuleModal() {
    moduleModal.classList.remove('active');
}

// Make closeModuleModal available globally
window.closeModuleModal = closeModuleModal;

// Initialize module forms
function initModuleForms() {
    // Disease form
    const diseaseForm = document.getElementById('diseaseForm');
    if (diseaseForm) {
        diseaseForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const imageFile = document.getElementById('diseaseImage').files[0];
            formData.append('image', imageFile);
            
            showLoading('diseaseResult');
            
            try {
                const response = await fetch(`${API_BASE}/disease/predict`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                showDiseaseResult(data);
            } catch (error) {
                showError('diseaseResult', error);
            }
        });
    }
    
    // Crop form
    const cropForm = document.getElementById('cropForm');
    if (cropForm) {
        cropForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                nitrogen: parseInt(document.getElementById('cropN').value),
                phosphorus: parseInt(document.getElementById('cropP').value),
                potassium: parseInt(document.getElementById('cropK').value),
                ph: parseFloat(document.getElementById('cropPH').value),
                temperature: parseFloat(document.getElementById('cropTemp').value),
                humidity: parseFloat(document.getElementById('cropHumidity').value),
                rainfall: parseFloat(document.getElementById('cropRainfall').value)
            };
            
            showLoading('cropResult');
            
            try {
                const response = await fetch(`${API_BASE}/crop/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                showCropResult(data);
            } catch (error) {
                showError('cropResult', error);
            }
        });
    }
    
    // Fertilizer form
    const fertilizerForm = document.getElementById('fertilizerForm');
    if (fertilizerForm) {
        fertilizerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                crop: document.getElementById('fertCrop').value,
                soil_type: document.getElementById('fertSoil').value,
                nitrogen: parseInt(document.getElementById('fertN').value),
                phosphorus: parseInt(document.getElementById('fertP').value),
                potassium: parseInt(document.getElementById('fertK').value),
                moisture: parseInt(document.getElementById('fertMoisture').value)
            };
            
            showLoading('fertilizerResult');
            
            try {
                const response = await fetch(`${API_BASE}/fertilizer/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                showFertilizerResult(data);
            } catch (error) {
                showError('fertilizerResult', error);
            }
        });
    }
    
    // Weather form
    const weatherForm = document.getElementById('weatherForm');
    if (weatherForm) {
        weatherForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const location = document.getElementById('weatherLocation').value;
            
            showLoading('weatherResult');
            
            try {
                const response = await fetch(`${API_BASE}/weather/current?location=${encodeURIComponent(location)}`);
                const data = await response.json();
                showWeatherResult(data);
            } catch (error) {
                showError('weatherResult', error);
            }
        });
    }
    
    // Yield form
    const yieldForm = document.getElementById('yieldForm');
    if (yieldForm) {
        yieldForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                crop: document.getElementById('yieldCrop').value,
                area: parseFloat(document.getElementById('yieldArea').value),
                region: document.getElementById('yieldRegion').value,
                season: document.getElementById('yieldSeason').value
            };
            
            showLoading('yieldResult');
            
            try {
                const response = await fetch(`${API_BASE}/yield/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                showYieldResult(data);
            } catch (error) {
                showError('yieldResult', error);
            }
        });
    }
    
    // Soil form
    const soilForm = document.getElementById('soilForm');
    if (soilForm) {
        soilForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const imageFile = document.getElementById('soilImage').files[0];
            formData.append('image', imageFile);
            
            showLoading('soilResult');
            
            try {
                const response = await fetch(`${API_BASE}/soil/classify`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                showSoilResult(data);
            } catch (error) {
                showError('soilResult', error);
            }
        });
    }
    
    // Risk form
    const riskForm = document.getElementById('riskForm');
    if (riskForm) {
        riskForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                crop: document.getElementById('riskCrop').value,
                coverage: parseFloat(document.getElementById('riskCoverage').value),
                weather_risk: parseFloat(document.getElementById('riskWeather').value),
                crop_success_rate: parseFloat(document.getElementById('riskSuccess').value)
            };
            
            showLoading('riskResult');
            
            try {
                const response = await fetch(`${API_BASE}/risk/calculate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                showRiskResult(data);
            } catch (error) {
                showError('riskResult', error);
            }
        });
    }
    
    // Irrigation form
    const irrigationForm = document.getElementById('irrigationForm');
    if (irrigationForm) {
        irrigationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                soil_moisture: parseFloat(document.getElementById('irrMoisture').value),
                temperature: parseFloat(document.getElementById('irrTemp').value),
                crop_type: document.getElementById('irrCrop').value,
                growth_stage: document.getElementById('irrStage').value
            };
            
            showLoading('irrigationResult');
            
            try {
                const response = await fetch(`${API_BASE}/irrigation/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                showIrrigationResult(data);
            } catch (error) {
                showError('irrigationResult', error);
            }
        });
    }
}

// Result display functions
function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="result-card">
            <div class="loading" style="justify-content: center;">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
}

function showError(elementId, error) {
    document.getElementById(elementId).innerHTML = `
        <div class="result-card">
            <p style="color: #dc2626;">Error: ${error.message}</p>
        </div>
    `;
}

function showDiseaseResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('diseaseResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">${result.disease || 'Detection Result'}</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${(result.confidence * 100).toFixed(1)}%</span>
                        <span class="result-label">CONFIDENCE</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.is_healthy ? '✅' : '⚠️'}</span>
                        <span class="result-label">STATUS</span>
                    </div>
                </div>
                ${result.treatment ? `<p style="margin-top: 1rem;">${result.treatment}</p>` : ''}
            </div>
        `;
    } else {
        showError('diseaseResult', new Error(data.message));
    }
}

function showCropResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('cropResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">Recommended Crops</h3>
                <div class="result-grid">
                    ${result.recommendations ? result.recommendations.slice(0, 3).map((rec, i) => `
                        <div class="result-item">
                            <span class="result-value">${rec.crop || rec}</span>
                            <span class="result-label">#${i + 1}</span>
                        </div>
                    `).join('') : '<div class="result-item"><span class="result-value">' + result.recommended_crop + '</span></div>'}
                </div>
            </div>
        `;
    } else {
        showError('cropResult', new Error(data.message));
    }
}

function showFertilizerResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('fertilizerResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">${result.recommended_fertilizer}</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${result.npk_ratio?.nitrogen || 0}-${result.npk_ratio?.phosphorus || 0}-${result.npk_ratio?.potassium || 0}</span>
                        <span class="result-label">NPK RATIO</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${((result.confidence || 0.7) * 100).toFixed(0)}%</span>
                        <span class="result-label">CONFIDENCE</span>
                    </div>
                </div>
                <p style="margin-top: 1rem;">${result.application_notes || ''}</p>
            </div>
        `;
    } else {
        showError('fertilizerResult', new Error(data.message));
    }
}

function showWeatherResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('weatherResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">${result.location}</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${result.current?.temperature || 'N/A'}°C</span>
                        <span class="result-label">TEMPERATURE</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.current?.humidity || 'N/A'}%</span>
                        <span class="result-label">HUMIDITY</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.current?.condition || 'N/A'}</span>
                        <span class="result-label">CONDITION</span>
                    </div>
                </div>
                ${result.agricultural_insights ? `
                    <p style="margin-top: 1rem;">
                        <strong>Irrigation needed:</strong> ${result.agricultural_insights.irrigation_needed ? 'Yes' : 'No'}<br>
                        <strong>Spraying possible:</strong> ${result.agricultural_insights.spraying_possible ? 'Yes' : 'No'}
                    </p>
                ` : ''}
            </div>
        `;
    } else {
        showError('weatherResult', new Error(data.message));
    }
}

function showYieldResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('yieldResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">Yield Prediction</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${result.predicted_yield || 'N/A'}</span>
                        <span class="result-label">PREDICTED YIELD</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.yield_per_hectare || 'N/A'}</span>
                        <span class="result-label">PER HECTARE</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${((result.confidence || 0.8) * 100).toFixed(0)}%</span>
                        <span class="result-label">CONFIDENCE</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        showError('yieldResult', new Error(data.message));
    }
}

function showSoilResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('soilResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">${result.soil_type}</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${((result.confidence || 0.8) * 100).toFixed(0)}%</span>
                        <span class="result-label">CONFIDENCE</span>
                    </div>
                </div>
                ${result.recommended_crops ? `<p style="margin-top: 1rem;"><strong>Recommended crops:</strong> ${result.recommended_crops.join(', ')}</p>` : ''}
            </div>
        `;
    } else {
        showError('soilResult', new Error(data.message));
    }
}

function showRiskResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('riskResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">Risk Assessment</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${result.ars_score || 'N/A'}</span>
                        <span class="result-label">ARS SCORE</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.risk_category || 'Moderate'}</span>
                        <span class="result-label">RISK LEVEL</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">₹${result.insurance_premium || 'N/A'}</span>
                        <span class="result-label">PREMIUM</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        showError('riskResult', new Error(data.message));
    }
}

function showIrrigationResult(data) {
    if (data.success) {
        const result = data.data;
        document.getElementById('irrigationResult').innerHTML = `
            <div class="result-card">
                <h3 class="result-title">Irrigation Recommendation</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-value">${result.water_required || 'N/A'}</span>
                        <span class="result-label">WATER NEEDED</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.irrigation_needed ? 'Yes' : 'No'}</span>
                        <span class="result-label">IRRIGATION</span>
                    </div>
                    <div class="result-item">
                        <span class="result-value">${result.best_time || 'Morning'}</span>
                        <span class="result-label">BEST TIME</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        showError('irrigationResult', new Error(data.message));
    }
}

// ============================================
// Smooth Scroll
// ============================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Add visible class to cards in viewport on load
    document.querySelectorAll('.module-card').forEach(card => {
        const rect = card.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
            card.classList.add('visible');
        }
    });
});
