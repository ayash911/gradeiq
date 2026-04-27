document.addEventListener('DOMContentLoaded', () => { // start the app!

    function initModels() { // wake up the engines
        fetch('/api/grade-manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_answer: "init",
                max_marks: 1,
                student_answers: { "init_student": "init" }
            })
        })
            .then(res => res.json())
            .then(() => {
                document.getElementById('init-loader').classList.add('hidden'); // hide loader when ready
            })
            .catch(err => {
                console.error("Init error", err);
                document.getElementById('init-loader').classList.add('hidden'); // hide loader anyway
            });
    }

    initModels(); // run init

    const inputView = document.getElementById('input-view'); // main views
    const progressView = document.getElementById('progress-view');
    const resultsView = document.getElementById('results-view');
    const submitBtn = document.getElementById('submit-btn');
    const submitSpinner = document.getElementById('submit-spinner');

    let currentSessionId = null; // tracking session
    let currentMode = 'zip'; // mode selector
    const tabs = document.querySelectorAll('.tab');
    const zipGroup = document.getElementById('zip-input-group');
    const manualGroup = document.getElementById('manual-input-group');

    tabs.forEach(tab => { // handle tab clicks
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentMode = tab.dataset.mode;

            if (currentMode === 'zip') {
                zipGroup.classList.remove('hidden');
                manualGroup.classList.add('hidden');
            } else {
                zipGroup.classList.add('hidden');
                manualGroup.classList.remove('hidden');
            }
        });
    });

    const dropZone = document.getElementById('drop-zone'); // file upload bits
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');

    dropZone.addEventListener('click', () => fileInput.click()); // click to upload

    dropZone.addEventListener('dragover', (e) => { // drag stuff
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
    });
    dropZone.addEventListener('dragleave', (e) => { // leave drag
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
    });
    dropZone.addEventListener('drop', (e) => { // drop file
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            updateFileDisplay();
        }
    });

    fileInput.addEventListener('change', updateFileDisplay); // file changed
    removeFileBtn.addEventListener('click', () => { // clear file
        fileInput.value = '';
        updateFileDisplay();
    });

    function updateFileDisplay() { // show file name
        if (fileInput.files.length > 0) {
            fileName.textContent = fileInput.files[0].name;
            dropZone.classList.add('hidden');
            fileInfo.classList.remove('hidden');
        } else {
            dropZone.classList.remove('hidden');
            fileInfo.classList.add('hidden');
        }
    }

    document.getElementById('demo-btn').addEventListener('click', () => { // load demo data
        document.getElementById('model-answer').value = "Mitochondria is an organelle in the cell. It produces energy for the cell in the form of ATP through cellular respiration.";
        document.getElementById('max-marks').value = "5";
        tabs[1].click(); // switch to manual mode
        document.getElementById('manual-answers').value = "Student_A: Mitochondria is the powerhouse of the cell and makes ATP energy.\nStudent_B: Mitochondria is a type of cell that does not produce any energy. The nucleus produces ATP.\nStudent_C: Mitochondria produces energy using glucose and oxygen in a process called respiration.";
        showToast("Demo configuration loaded.");
    });

    document.getElementById('new-assessment-btn').addEventListener('click', () => { // start fresh
        resultsView.classList.add('hidden');
        inputView.classList.remove('hidden');
        document.getElementById('results-grid').innerHTML = ''; // clear results
        currentSessionId = null;
    });

    document.getElementById('grading-form').addEventListener('submit', (e) => { // handle submit
        e.preventDefault();
        const modelAnswer = document.getElementById('model-answer').value;
        const maxMarks = document.getElementById('max-marks').value;
        const isDebug = document.getElementById('debug-mode-check').checked;

        if (currentMode === 'zip') { // zip mode
            if (fileInput.files.length === 0) {
                showToast('Please upload a ZIP file first', 'error');
                return;
            }
            const formData = new FormData();
            formData.append('model_answer', modelAnswer);
            formData.append('max_marks', maxMarks);
            formData.append('answer_sheets', fileInput.files[0]);
            formData.append('debug', isDebug);
            submitData('/api/grade', formData, true);
        } else { // manual mode
            const manualText = document.getElementById('manual-answers').value;
            if (!manualText.trim()) {
                showToast('Please enter some student answers', 'error');
                return;
            }
            const studentAnswers = {};
            manualText.split('\n').forEach((line, i) => {
                if (line.trim()) {
                    let key = `Student_${i + 1}`;
                    let text = line.trim();
                    if (text.includes(':')) {
                        key = text.substring(0, text.indexOf(':')).trim();
                        text = text.substring(text.indexOf(':') + 1).trim();
                    }
                    studentAnswers[key] = text;
                }
            });

            submitData('/api/grade-manual', JSON.stringify({
                model_answer: modelAnswer,
                max_marks: maxMarks,
                student_answers: studentAnswers
            }), false);
        }
    });

    function submitData(url, data, isMultipart) { // send data to server
        inputView.classList.add('hidden');
        resultsView.classList.add('hidden');
        progressView.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = "Processing...";
        submitSpinner.classList.remove('hidden');

        const options = { method: 'POST', body: data };
        if (!isMultipart) {
            options.headers = { 'Content-Type': 'application/json' };
        }

        fetch(url, options)
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                renderResults(data);
            })
            .catch(err => {
                showToast(err.message, 'error');
                progressView.classList.add('hidden');
                inputView.classList.remove('hidden');
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.querySelector('.btn-text').textContent = "Run Assessment";
                submitSpinner.classList.add('hidden');
            });
    }

    function renderResults(data) { // show the scores
        progressView.classList.add('hidden');
        resultsView.classList.remove('hidden');
        currentSessionId = data.session_id;
        document.getElementById('results-summary').textContent = `Processed ${data.results.length} submissions successfully.`;

        const grid = document.getElementById('results-grid');
        grid.innerHTML = '';

        data.results.forEach(r => {
            const card = document.createElement('div');
            card.className = 'result-item';

            let badgeClass = 'badge-neutral';
            if (r.nli_label === 'CORRECT') badgeClass = 'badge-correct';
            if (r.nli_label === 'PARTIAL') badgeClass = 'badge-partial';
            if (r.nli_label === 'CONTRADICTION') badgeClass = 'badge-contradiction';

            let visionBtnHtml = '';
            if (r.image_b64) { // vision diagnostic btn
                const modalData = encodeURIComponent(JSON.stringify({
                    img: r.image_b64,
                    boxes: r.debug_ocr || []
                }));
                visionBtnHtml = `<button class="btn-vision" data-vision="${modalData}">View Scan Diagnostics</button>`;
            }

            card.innerHTML = `
                <div class="ri-header">
                    <div>
                        <div class="ri-student">${escapeHtml(r.student)}</div>
                        <div class="badge ${badgeClass}">${r.nli_label}</div>
                    </div>
                    <div class="ri-score">
                        ${r.suggested_marks}<span>/${r.max_marks}</span>
                    </div>
                </div>
                <div class="ri-stats">
                    <div class="ri-stat">
                        <span class="ri-stat-label">Similarity</span>
                        <span class="ri-stat-val">${(r.similarity * 100).toFixed(0)}%</span>
                    </div>
                    <div class="ri-stat">
                        <span class="ri-stat-label">Confidence</span>
                        <span class="ri-stat-val">${r.confidence.split(' ')[0]}</span>
                    </div>
                </div>
                <div class="ri-text-box">
                    <div class="ri-text-label">Extracted Text</div>
                    ${escapeHtml(r.extracted_text || 'No text extracted.')}
                </div>
                ${visionBtnHtml}
            `;
            grid.appendChild(card);
        });

        document.querySelectorAll('.btn-vision').forEach(btn => { // open diagnostic modal
            btn.addEventListener('click', (e) => {
                const data = JSON.parse(decodeURIComponent(e.target.getAttribute('data-vision')));
                openVisionModal(data);
            });
        });
    }

    const modal = document.getElementById('vision-modal'); // modal handling
    const modalClose = document.getElementById('modal-close');
    const visionContainer = document.getElementById('vision-container');

    function openVisionModal(data) { // show image with ocr boxes
        let html = `<div class="vision-viewport">
                        <img src="data:image/jpeg;base64,${data.img}" class="vision-img">`;

        data.boxes.forEach(line => {
            const [x1, y1, x2, y2] = line.box;
            const left = (x1 * 100).toFixed(2);
            const top = (y1 * 100).toFixed(2);
            const width = ((x2 - x1) * 100).toFixed(2);
            const height = ((y2 - y1) * 100).toFixed(2);
            const struckClass = line.is_struck ? 'struck' : '';

            html += `<div class="ocr-box ${struckClass}" 
                          style="left:${left}%; top:${top}%; width:${width}%; height:${height}%;"
                          data-text="${escapeHtml(line.text)}"></div>`;
        });
        html += `</div>`;

        visionContainer.innerHTML = html;
        modal.classList.remove('hidden');
        modal.classList.add('active');
    }

    modalClose.addEventListener('click', () => { // close modal
        modal.classList.remove('active');
        setTimeout(() => modal.classList.add('hidden'), 200);
    });

    document.getElementById('download-btn').addEventListener('click', () => { // download csv
        if (!currentSessionId) return;
        window.location.href = `/api/download/${currentSessionId}`;
    });

    function escapeHtml(unsafe) { // helper for safety
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function showToast(message, type = 'success') { // helper for notifications
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type === 'error' ? 'error' : ''}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});
