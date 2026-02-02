// Gait Analysis Medical Dashboard Script

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMsg = document.getElementById('error-msg');
const videoContainer = document.getElementById('video-container');
const videoPreview = document.getElementById('video-preview');

// Chart Instances
let kneeChart = null;
let hipChart = null;

// Initialize
function init() {
    checkModelStatus();
    setupDragAndDrop();
}

function checkModelStatus() {
    fetch('/api/model-status')
        .then(response => response.json())
        .then(data => {
            const dot = document.getElementById('model-status-dot');
            const text = document.getElementById('model-status-text');

            if (data.model_loaded) {
                dot.style.backgroundColor = '#4CAF50'; // Green
                text.textContent = `System Online (${data.type})`;
                text.style.color = '#4CAF50';
            } else {
                dot.style.backgroundColor = '#f44336'; // Red
                text.textContent = 'System Offline (Model not loaded)';
                text.style.color = '#f44336';
            }
        })
        .catch(err => {
            console.error('Status check failed:', err);
            document.getElementById('model-status-text').textContent = 'Server Unreachable';
        });
}

function setupDragAndDrop() {
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', handleFileSelect);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
        dropZone.style.background = 'rgba(67, 97, 238, 0.1)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background = 'rgba(255, 255, 255, 0.03)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.background = 'rgba(255, 255, 255, 0.03)';

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

function handleFileSelect(e) {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // Validation
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    // Minimal validation as mime types vary

    // Show Preview
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    videoContainer.style.display = 'block';

    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('video', file);

    // UI State
    loading.style.display = 'flex';
    results.style.display = 'none';
    errorMsg.style.display = 'none';
    document.querySelector('.upload-area').style.display = 'none';

    // Determine progress simulation (fake it for UX)
    let progress = 0;
    const interval = setInterval(() => {
        progress += 5;
        if (progress > 90) clearInterval(interval);
    }, 500);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            loading.style.display = 'none';

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error);
            }
        })
        .catch(err => {
            clearInterval(interval);
            loading.style.display = 'none';
            showError('Upload failed. Please check server connection.');
            console.error(err);
        });
}

function displayResults(data) {
    results.style.display = 'grid';

    // 1. Classification
    const predText = document.getElementById('prediction-text');
    predText.textContent = data.prediction.toUpperCase().replace('_', ' ');

    // Color coding
    predText.className = '';
    if (data.prediction === 'normal') predText.classList.add('status-normal');
    else if (data.prediction === 'ataxic' || data.prediction === 'parkinsonian') predText.classList.add('status-critical');
    else predText.classList.add('status-warning');

    // Confidence
    const confVal = Math.round(data.confidence * 100);
    document.getElementById('confidence-val').textContent = confVal;
    document.getElementById('confidence-fill').style.width = `${confVal}%`;

    // 2. Metrics (New)
    if (data.metrics) {
        document.getElementById('cadence-val').textContent = data.metrics.cadence || '--';
        document.getElementById('symmetry-val').textContent = data.metrics.symmetry || '--';
        document.getElementById('width-val').textContent = data.metrics.step_width || '--';
        document.getElementById('velocity-val').textContent = data.metrics.velocity || 'N/A';
    }

    // 3. Probabilities list
    const probList = document.getElementById('prob-list');
    probList.innerHTML = '';

    // Sort
    const sortedProbs = Object.entries(data.all_probabilities)
        .sort(([, a], [, b]) => b - a);

    sortedProbs.forEach(([label, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        const percent = Math.round(prob * 100);

        item.innerHTML = `
            <div class="prob-label">
                <span>${label.replace('_', ' ')}</span>
                <span>${percent}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${percent}%"></div>
            </div>
        `;
        probList.appendChild(item);
    });

    // 4. Render Charts (New)
    if (data.timeseries) {
        renderCharts(data.timeseries);
    }
}

function renderCharts(tsData) {
    // Destroy previous charts if existing
    if (kneeChart) kneeChart.destroy();
    if (hipChart) hipChart.destroy();

    const frames = tsData.frames;

    // Knee Chart
    const ctxKnee = document.getElementById('kneeChart').getContext('2d');
    kneeChart = new Chart(ctxKnee, {
        type: 'line',
        data: {
            labels: frames,
            datasets: [
                {
                    label: 'Left Knee',
                    data: tsData.l_knee,
                    borderColor: '#4361ee',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Right Knee',
                    data: tsData.r_knee,
                    borderColor: '#f72585',
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: getChartOptions('Knee Flexion Angle (deg)')
    });

    // Hip Chart
    const ctxHip = document.getElementById('hipChart').getContext('2d');
    hipChart = new Chart(ctxHip, {
        type: 'line',
        data: {
            labels: frames,
            datasets: [
                {
                    label: 'Left Hip',
                    data: tsData.l_hip,
                    borderColor: '#4361ee',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Right Hip',
                    data: tsData.r_hip,
                    borderColor: '#f72585',
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: getChartOptions('Hip Angle (deg)')
    });
}

function getChartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#ccc' }
            }
        },
        scales: {
            y: {
                title: { display: true, text: yLabel, color: '#888' },
                grid: { color: '#333' },
                ticks: { color: '#aaa' }
            },
            x: {
                display: false // Hide frame numbers to reduce clutter
            }
        },
        animation: {
            duration: 1000
        }
    };
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.style.display = 'block';
}

// Start
init();

// ==========================================
// PDF REPORT GENERATION
// ==========================================
async function generatePDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // 1. Header
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(40, 40, 40);
    doc.text("Gait Analysis Assessment Report", 20, 20);

    // 2. Patient Info (Date)
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 28);

    // Draw Line
    doc.setDrawColor(200, 200, 200);
    doc.line(20, 32, pageWidth - 20, 32);

    // 3. Clinical Verdict
    const verdict = document.getElementById('prediction-text').innerText;
    const confidence = document.getElementById('confidence-val').innerText;

    doc.setFontSize(16);
    doc.setTextColor(0, 0, 0);
    doc.text(`Diagnosis: ${verdict}`, 20, 45);

    doc.setFontSize(12);
    doc.setTextColor(60, 60, 60);
    doc.text(`AI Confidence: ${confidence}%`, 20, 52);

    // 4. Clinical Metrics Table
    doc.text("Quantitative Metrics:", 20, 65);

    const cadence = document.getElementById('cadence-val').innerText;
    const sym = document.getElementById('symmetry-val').innerText;
    const width = document.getElementById('width-val').innerText;
    const vel = document.getElementById('velocity-val').innerText;

    const metricsData = [
        [`Cadence`, `${cadence} steps/min`],
        [`Symmetry Index`, `${sym} (L/R)`],
        [`Step Width`, `${width} px`],
        [`Velocity`, `${vel}`]
    ];

    let yPos = 72;
    doc.setFontSize(11);
    metricsData.forEach(row => {
        doc.text(`• ${row[0]}: ${row[1]}`, 25, yPos);
        yPos += 7;
    });

    // 5. Clinician Notes
    const severity = document.getElementById('severity-select');
    let grade = "Not Selected";
    if (severity.selectedIndex > 0) {
        grade = severity.options[severity.selectedIndex].text;
    }
    const notes = document.getElementById('clinical-notes').value || "No notes entered.";

    yPos += 10;
    doc.setFont("helvetica", "bold");
    doc.text("Clinician Assessment:", 20, yPos);

    yPos += 7;
    doc.setFont("helvetica", "normal");
    doc.text(`Severity: ${grade}`, 25, yPos);

    yPos += 7;
    doc.text("Observations:", 25, yPos);
    yPos += 6;

    const splitNotes = doc.splitTextToSize(notes, pageWidth - 50);
    doc.text(splitNotes, 25, yPos);

    yPos += (splitNotes.length * 6) + 10;

    // 6. Kinematic Graphs (Capture HTML Canvas)
    if (yPos > 200) { doc.addPage(); yPos = 20; }

    doc.setFont("helvetica", "bold");
    doc.text("Kinematic Analysis (Knee/Hip):", 20, yPos);
    yPos += 10;

    try {
        const kneeCanvas = document.getElementById('kneeChart');
        const hipCanvas = document.getElementById('hipChart');

        // Convert to Image
        // Scale down to fit side-by-side
        const imgWidth = (pageWidth - 50) / 2;
        const imgHeight = 60; // Aspect ratio approx

        const kneeImg = kneeCanvas.toDataURL("image/png");
        const hipImg = hipCanvas.toDataURL("image/png");

        doc.addImage(kneeImg, 'PNG', 20, yPos, imgWidth, imgHeight);
        doc.addImage(hipImg, 'PNG', 20 + imgWidth + 10, yPos, imgWidth, imgHeight);

    } catch (e) {
        console.error("Chart capture failed", e);
        doc.text("Error capturing charts.", 20, yPos + 10);
    }

    // Save
    doc.save("Gait_Assessment_Report.pdf");
}

function resetAnalysis() {
    // Hide Results
    results.style.display = 'none';

    // Show Upload
    document.querySelector('.upload-area').style.display = 'flex';
    document.querySelector('.top-section').style.display = 'block';

    // Reset Video Preview
    videoContainer.style.display = 'none';
    videoPreview.src = '';

    // Clear Input
    fileInput.value = '';

    // Reset Status
    const statusText = document.getElementById('prediction-text');
    statusText.textContent = '--';
    statusText.className = '';

    // Clear Charts (Optional, as renderCharts destroys old ones)
}
