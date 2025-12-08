const API_BASE_URL = 'http://127.0.0.1:5000';

document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const sections = document.querySelectorAll('.test-section');

    function switchTab(targetId) {
        tabs.forEach(tab => tab.classList.remove('border-indigo-600', 'text-indigo-600'));
        sections.forEach(section => section.classList.add('hidden'));

        const activeTab = document.querySelector(`[data-target="${targetId}"]`);
        const activeSection = document.getElementById(targetId);
        if (activeTab && activeSection) {
            activeTab.classList.add('border-indigo-600', 'text-indigo-600');
            activeSection.classList.remove('hidden');
        }
    }

    tabs.forEach(tab => tab.addEventListener('click', e => switchTab(e.target.dataset.target)));
    switchTab('predict-section');
});

function showLoading(buttonId, textId) {
    const btn = document.getElementById(buttonId);
    const textSpan = document.getElementById(textId);
    if (!btn || !textSpan) return;
    btn.dataset.originalText = textSpan.textContent;
    textSpan.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Processing...`;
    btn.disabled = true;
}

function hideLoading(buttonId, textId) {
    const btn = document.getElementById(buttonId);
    const textSpan = document.getElementById(textId);
    if (!btn || !textSpan) return;
    textSpan.textContent = btn.dataset.originalText || 'Run';
    btn.disabled = false;
    delete btn.dataset.originalText;
}

function displayResult(data) {
    const resultsCard = document.getElementById('results');
    const output = document.getElementById('result-output');
    resultsCard.classList.remove('hidden');

    // Persistent AI container
    let aiContainer = document.getElementById('ai-prediction-container');
    if (!aiContainer) {
        aiContainer = document.createElement('div');
        aiContainer.id = 'ai-prediction-container';
        output.appendChild(aiContainer);
    }

    // Update AI if present
    if (data.result && data.confidence) {
        aiContainer.innerHTML = `
            <h3 class="text-lg font-semibold">AI Prediction</h3>
            <div class="progress-labels flex justify-between mb-1">
                <span id="ai-result-text">${data.result}</span>
                <span id="ai-confidence-text">${data.confidence}</span>
            </div>
            <div class="progress-bar-container">
                <div id="ai-progress-bar" class="progress-bar ${data.result === 'REAL' ? 'bg-green' : 'bg-red'}" style="width:0%"></div>
            </div>
        `;
        const bar = document.getElementById('ai-progress-bar');
        bar.style.transition = 'width 1s ease-in-out';
        bar.style.width = parseFloat(data.confidence) + '%';
    }

    // Watermark embed response (download link)
    if (data.download_url || data.embedded_audio_base64) {
        let embedBox = document.getElementById('embed-result-box');
        if (!embedBox) {
            embedBox = document.createElement('div');
            embedBox.id = 'embed-result-box';
            output.appendChild(embedBox);
        }
        embedBox.innerHTML = `
            <h3 class="text-lg font-semibold">Watermark Embed</h3>
            <div class="mb-2">Filename: ${data.filename || '(unknown)'}</div>
            <div class="flex gap-2">
                <button id="downloadEmbeddedBtn" class="btn">Download Watermarked Audio</button>
            </div>
        `;

        const dlBtn = document.getElementById('downloadEmbeddedBtn');
        dlBtn.onclick = () => {
            // Prefer server download_url (deletes after download)
            if (data.download_url) {
                const a = document.createElement('a');
                a.href = data.download_url;
                a.download = data.filename || 'watermarked.wav';
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else if (data.embedded_audio_base64) {
                const byteChars = atob(data.embedded_audio_base64);
                const byteArr = new Uint8Array(byteChars.length);
                for (let i = 0; i < byteChars.length; i++) byteArr[i] = byteChars.charCodeAt(i);
                const blob = new Blob([byteArr], { type: 'audio/wav' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = data.filename || 'watermarked.wav';
                document.body.appendChild(a); a.click(); a.remove();
                URL.revokeObjectURL(url);
            }
        };
    }

    // Verification display (supports REAL/FAKE with confidence)
    if (data.verification_result) {
        let vBox = document.getElementById('verify-result-box');
        if (!vBox) {
            vBox = document.createElement('div');
            vBox.id = 'verify-result-box';
            output.appendChild(vBox);
        }

        const isReal = data.verification_result === 'REAL' || data.verification_result === 'SUCCESS';
        const conf = data.confidence || (data.confidence === 0 ? '0%' : '');
        vBox.innerHTML = `
            <h3 class="text-lg font-semibold">Watermark Verification</h3>
            <div class="progress-labels flex justify-between mb-1">
                <span id="verify-status-text">${data.verification_result}</span>
                <span id="verify-confidence-text">${conf}</span>
            </div>
            <div class="progress-bar-container">
                <div id="verify-progress-bar" class="progress-bar ${isReal ? 'bg-green' : 'bg-red'}" style="width:0%"></div>
            </div>
            <div class="mt-2">Extracted (preview): ${data.extracted_watermark_text ? data.extracted_watermark_text : (data.extracted_bits ? data.extracted_bits.slice(0, 32).join('') + '...' : 'N/A')}</div>
        `;
        const vb = document.getElementById('verify-progress-bar');
        // convert "xx.xx%" to number if necessary
        let pct = 0;
        if (typeof conf === 'string' && conf.endsWith('%')) pct = parseFloat(conf);
        else if (typeof conf === 'number') pct = conf;
        else pct = isReal ? 100 : 40;
        vb.style.transition = 'width 1s ease-in-out';
        vb.style.width = `${pct}%`;
    }

    // Errors
    if (data.error) {
        const errBox = document.createElement('div');
        errBox.className = 'text-red-600 font-semibold';
        errBox.textContent = `Error: ${data.error}`;
        output.appendChild(errBox);
    }
}

// Run predict or verify
async function runTest(endpoint, fileInputId, buttonId, textId) {
    const fileInput = document.getElementById(fileInputId);
    const file = fileInput.files[0];
    if (!file) { displayResult({ error: 'Please select a .wav file.' }); return; }
    showLoading(buttonId, textId);

    const formData = new FormData();
    formData.append('file', file);

    // For verify, we no longer send original watermark
    if (endpoint === '/verify') {
        // No extra data needed
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST', body: formData });
        const result = await response.json();
        displayResult(result);
    } catch (err) {
        displayResult({ error: 'API call failed', details: err.toString() });
    } finally {
        hideLoading(buttonId, textId);
    }
}

// Run embed
async function runEmbedTest() {
    const fileInput = document.getElementById('embedFile');
    const watermarkInput = document.getElementById('watermarkData');
    const file = fileInput.files[0];
    const watermark = watermarkInput.value;
    if (!file || !watermark) { displayResult({ error: 'Select audio and enter watermark text.' }); return; }

    const buttonId = 'embedButton';
    const textId = 'embed-text';
    showLoading(buttonId, textId);

    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('watermark_data', watermark);

    try {
        const response = await fetch(`${API_BASE_URL}/embed`, { method: 'POST', body: formData });
        const result = await response.json();
        displayResult(result);
    } catch (err) {
        displayResult({ error: 'Embed API failed', details: err.toString() });
    } finally {
        hideLoading(buttonId, textId);
    }
}
