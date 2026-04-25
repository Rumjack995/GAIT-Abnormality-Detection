// Gait Analysis Medical Dashboard — Multi-Model Script

const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const loading      = document.getElementById('loading');
const results      = document.getElementById('results');
const errorMsg     = document.getElementById('error-msg');
const videoContainer = document.getElementById('video-container');
const videoPreview = document.getElementById('video-preview');

let kneeChart = null;
let hipChart  = null;

// ── Init ────────────────────────────────────────────────────────────────────

function init() {
    checkModelStatus();
    setupDragAndDrop();
}

// ── Model Status ────────────────────────────────────────────────────────────

function checkModelStatus() {
    fetch('/api/model-status')
        .then(r => r.json())
        .then(data => {
            const dot  = document.getElementById('model-status-dot');
            const text = document.getElementById('model-status-text');

            // Update per-model badges
            const allLoaded = Object.values(data.models).every(m => m.loaded);
            const anyLoaded = Object.values(data.models).some(m => m.loaded);

            for (const [key, info] of Object.entries(data.models)) {
                const badge    = document.getElementById(`badge-${key}`);
                const radioBtn = document.getElementById(`model-${key}`);
                if (!badge) continue;

                if (info.loaded) {
                    badge.textContent = '✓ Ready';
                    badge.className   = 'mc-badge';
                    if (radioBtn) radioBtn.disabled = false;
                } else {
                    badge.textContent = '✗ Not Trained';
                    badge.className   = 'mc-badge not-loaded';
                    if (radioBtn) radioBtn.disabled = true;
                }
            }

            // Auto-select first loaded model
            for (const [key, info] of Object.entries(data.models)) {
                const radioBtn = document.getElementById(`model-${key}`);
                if (info.loaded && radioBtn) {
                    radioBtn.checked = true;
                    break;
                }
            }

            // Footer status
            if (anyLoaded) {
                const loadedNames = Object.values(data.models)
                    .filter(m => m.loaded).map(m => m.label).join(', ');
                dot.style.backgroundColor  = allLoaded ? '#4CAF50' : '#FFC107';
                text.style.color           = allLoaded ? '#4CAF50' : '#FFC107';
                text.textContent           = `System Online — ${loadedNames}`;
            } else {
                dot.style.backgroundColor  = '#f44336';
                text.style.color           = '#f44336';
                text.textContent           = 'No models loaded — run train_all_models.py';
            }
        })
        .catch(() => {
            document.getElementById('model-status-text').textContent = 'Server Unreachable';
        });
}

// ── Drag & Drop ─────────────────────────────────────────────────────────────

function setupDragAndDrop() {
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
        dropZone.style.background  = 'rgba(67,97,238,0.1)';
    });
    dropZone.addEventListener('dragleave', e => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background  = 'rgba(255,255,255,0.03)';
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background  = 'rgba(255,255,255,0.03)';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
}

function handleFileSelect(e) {
    if (e.target.files.length) handleFile(e.target.files[0]);
}

function handleFile(file) {
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    videoContainer.style.display = 'block';
    uploadFile(file);
}

// ── Upload ──────────────────────────────────────────────────────────────────

function getSelectedModel() {
    const radios = document.querySelectorAll('input[name="model"]');
    for (const r of radios) {
        if (r.checked && !r.disabled) return r.value;
    }
    return 'lstm';
}

function uploadFile(file) {
    const selectedModel = getSelectedModel();

    const formData = new FormData();
    formData.append('video', file);
    formData.append('model', selectedModel);

    // UI state changes
    loading.style.display = 'flex';
    results.style.display = 'none';
    errorMsg.style.display = 'none';
    document.getElementById('upload-panel').style.display = 'none';

    // Update loading text to show which model
    const modelLabel = document.getElementById(`badge-${selectedModel}`)?.closest('label')
                                ?.querySelector('.mc-name')?.textContent || selectedModel;
    const loadingText = document.getElementById('loading-text');
    if (loadingText) loadingText.textContent = `Analysing with ${modelLabel}...`;

    fetch('/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            loading.style.display = 'none';
            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Analysis failed.');
                document.getElementById('upload-panel').style.display = 'block';
            }
        })
        .catch(err => {
            loading.style.display = 'none';
            showError('Upload failed. Please check server connection.');
            document.getElementById('upload-panel').style.display = 'block';
            console.error(err);
        });
}

// ── Display Results ─────────────────────────────────────────────────────────

function displayResults(data) {
    results.style.display = 'grid';

    // Model used badge
    const modelLabel = document.getElementById('model-used-label');
    if (modelLabel) modelLabel.textContent = data.model_used || '—';

    // Classification
    const predText = document.getElementById('prediction-text');
    predText.textContent = data.prediction.toUpperCase().replace(/_/g, ' ');
    predText.className = '';
    if (data.prediction === 'normal')           predText.classList.add('status-normal');
    else if (['ataxic','parkinsonian'].includes(data.prediction)) predText.classList.add('status-critical');
    else                                         predText.classList.add('status-warning');

    // Confidence
    const confVal = Math.round(data.confidence * 100);
    document.getElementById('confidence-val').textContent = confVal;
    document.getElementById('confidence-fill').style.width = `${confVal}%`;

    // Metrics
    if (data.metrics) {
        document.getElementById('cadence-val').textContent  = data.metrics.cadence   || '--';
        document.getElementById('symmetry-val').textContent = data.metrics.symmetry  || '--';
        document.getElementById('width-val').textContent    = data.metrics.step_width || '--';
        document.getElementById('velocity-val').textContent = data.metrics.velocity  || 'N/A';
    }

    // Probabilities
    const probList = document.getElementById('prob-list');
    probList.innerHTML = '';
    const sorted = Object.entries(data.all_probabilities).sort(([,a],[,b]) => b - a);
    sorted.forEach(([label, prob]) => {
        const pct  = Math.round(prob * 100);
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-label">
                <span>${label.replace(/_/g, ' ')}</span>
                <span>${pct}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${pct}%"></div>
            </div>
        `;
        probList.appendChild(item);
    });

    // Charts
    if (data.timeseries) renderCharts(data.timeseries);
}

// ── Charts ──────────────────────────────────────────────────────────────────

function renderCharts(tsData) {
    if (kneeChart) kneeChart.destroy();
    if (hipChart)  hipChart.destroy();

    const frames = tsData.frames;

    kneeChart = new Chart(document.getElementById('kneeChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: frames,
            datasets: [
                { label: 'Left Knee',  data: tsData.l_knee, borderColor: '#4361ee', borderWidth: 2, pointRadius: 0 },
                { label: 'Right Knee', data: tsData.r_knee, borderColor: '#f72585', borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: getChartOptions('Knee Flexion Angle (deg)')
    });

    hipChart = new Chart(document.getElementById('hipChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: frames,
            datasets: [
                { label: 'Left Hip',  data: tsData.l_hip, borderColor: '#4361ee', borderWidth: 2, pointRadius: 0 },
                { label: 'Right Hip', data: tsData.r_hip, borderColor: '#f72585', borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: getChartOptions('Hip Angle (deg)')
    });
}

function getChartOptions(yLabel) {
    return {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#ccc' } } },
        scales: {
            y: { title: { display: true, text: yLabel, color: '#888' }, grid: { color: '#333' }, ticks: { color: '#aaa' } },
            x: { display: false }
        },
        animation: { duration: 1000 }
    };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function showError(msg) {
    errorMsg.textContent   = '⚠️ ' + msg;
    errorMsg.style.display = 'block';
}

function resetAnalysis() {
    results.style.display = 'none';
    errorMsg.style.display = 'none';
    document.getElementById('upload-panel').style.display = 'block';
    videoContainer.style.display = 'none';
    videoPreview.src  = '';
    fileInput.value   = '';
    document.getElementById('prediction-text').textContent = '--';
    document.getElementById('prediction-text').className = '';
}

// ── PDF ─────────────────────────────────────────────────────────────────────

async function generatePDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const pw  = doc.internal.pageSize.getWidth();

    doc.setFont('helvetica','bold'); doc.setFontSize(22); doc.setTextColor(40,40,40);
    doc.text('Gait Analysis Assessment Report', 20, 20);

    doc.setFont('helvetica','normal'); doc.setFontSize(10); doc.setTextColor(100,100,100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 28);

    const modelUsed = document.getElementById('model-used-label')?.textContent || '—';
    doc.text(`AI Model: ${modelUsed}`, 20, 34);

    doc.setDrawColor(200,200,200); doc.line(20, 38, pw - 20, 38);

    const verdict    = document.getElementById('prediction-text').innerText;
    const confidence = document.getElementById('confidence-val').innerText;
    doc.setFontSize(16); doc.setTextColor(0,0,0);
    doc.text(`Diagnosis: ${verdict}`, 20, 50);
    doc.setFontSize(12); doc.setTextColor(60,60,60);
    doc.text(`AI Confidence: ${confidence}%`, 20, 58);

    doc.text('Quantitative Metrics:', 20, 72);
    const metrics = [
        ['Cadence',        document.getElementById('cadence-val').innerText + ' steps/min'],
        ['Symmetry Index', document.getElementById('symmetry-val').innerText + ' (L/R)'],
        ['Step Width',     document.getElementById('width-val').innerText],
        ['Velocity',       document.getElementById('velocity-val').innerText],
    ];
    let y = 80;
    metrics.forEach(([k,v]) => { doc.text(`• ${k}: ${v}`, 25, y); y += 7; });

    const severity = document.getElementById('severity-select');
    const grade    = severity.selectedIndex > 0 ? severity.options[severity.selectedIndex].text : 'Not Selected';
    const notes    = document.getElementById('clinical-notes').value || 'No notes entered.';

    y += 8;
    doc.setFont('helvetica','bold'); doc.text('Clinician Assessment:', 20, y); y += 7;
    doc.setFont('helvetica','normal');
    doc.text(`Severity: ${grade}`, 25, y); y += 7;
    doc.text('Observations:', 25, y); y += 6;
    const splitNotes = doc.splitTextToSize(notes, pw - 50);
    doc.text(splitNotes, 25, y); y += (splitNotes.length * 6) + 12;

    if (y > 200) { doc.addPage(); y = 20; }
    doc.setFont('helvetica','bold'); doc.text('Kinematic Analysis:', 20, y); y += 10;

    try {
        const iw = (pw - 50) / 2;
        doc.addImage(document.getElementById('kneeChart').toDataURL('image/png'), 'PNG', 20,       y, iw, 60);
        doc.addImage(document.getElementById('hipChart') .toDataURL('image/png'), 'PNG', 20+iw+10, y, iw, 60);
    } catch (e) { doc.text('Chart capture failed.', 20, y + 10); }

    doc.save('Gait_Assessment_Report.pdf');
}

// ── Start ────────────────────────────────────────────────────────────────────
init();
