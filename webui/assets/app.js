// file: webui/assets/app.js
let gReport = null;
let gFilteredPairs = [];
let distChart = null, donutChart = null;
let currentPairIdx = -1;

const $ = id => document.getElementById(id);
const esc = s => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// --- 1. Navigation & Init ---
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
        $(`tab-${btn.dataset.tab}`).classList.add('active');
    });
});

$('btnDark').onclick = () => {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
    if (distChart) distChart.options.scales.x.grid.color = isDark ? '#e5e7eb' : '#334155'; // Trick to update chart theme
};

// --- 2. Charting & Visualization (The "Google" Part) ---
function renderCharts(report) {
    // A. Histogram: Score Distribution
    const scores = report.pairs.map(p => p.score);
    const bins = [0, 0, 0, 0, 0]; // <0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, >0.9
    scores.forEach(s => {
        if (s < 0.6) bins[0]++;
        else if (s < 0.7) bins[1]++;
        else if (s < 0.8) bins[2]++;
        else if (s < 0.9) bins[3]++;
        else bins[4]++;
    });

    const ctxDist = $('distChart').getContext('2d');
    if (distChart) distChart.destroy();
    distChart = new Chart(ctxDist, {
        type: 'bar',
        data: {
            labels: ['< 60%', '60-70%', '70-80%', '80-90%', '> 90%'],
            datasets: [{
                label: 'Số lượng đoạn trùng',
                data: bins,
                backgroundColor: ['#e2e8f0', '#fef08a', '#fed7aa', '#fb7185', '#ef4444'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // B. Donut: Overall Similarity
    // Tính tổng số ký tự trùng / tổng ký tự document A
    let docLen = 0;
    const coveredIntervals = [];
    report.pairs.forEach(p => {
        const docSpan = p.a.char_span_in_doc;
        if (!docSpan) return;
        docLen = Math.max(docLen, docSpan[1]);
        // Merge các span con
        p.a.spans.forEach(([s, e]) => {
            coveredIntervals.push([docSpan[0] + s, docSpan[0] + e]);
        });
    });
    
    // Union intervals logic
    coveredIntervals.sort((a, b) => a[0] - b[0]);
    let merged = [];
    if (coveredIntervals.length > 0) {
        let [cs, ce] = coveredIntervals[0];
        for (let i = 1; i < coveredIntervals.length; i++) {
            let [ns, ne] = coveredIntervals[i];
            if (ns <= ce) ce = Math.max(ce, ne);
            else { merged.push([cs, ce]); cs = ns; ce = ne; }
        }
        merged.push([cs, ce]);
    }
    
    const coveredChars = merged.reduce((acc, [s, e]) => acc + (e - s), 0);
    const pct = docLen > 0 ? (coveredChars / docLen * 100) : 0;

    // Update Metrics text
    $('statPairs').innerText = report.pairs.length;
    $('wordsText').innerText = `~${Math.round(coveredChars / 5)} từ`;
    $('pctText').innerText = `${pct.toFixed(1)}%`;
    $('coverText').innerText = `${coveredChars.toLocaleString()} / ${docLen.toLocaleString()} chars`;

    const ctxDonut = $('donutScore').getContext('2d');
    if (donutChart) donutChart.destroy();
    donutChart = new Chart(ctxDonut, {
        type: 'doughnut',
        data: {
            labels: ['Trùng', 'Sạch'],
            datasets: [{
                data: [pct, 100 - pct],
                backgroundColor: ['#ef4444', '#e2e8f0'],
                borderWidth: 0
            }]
        },
        options: { cutout: '75%', plugins: { legend: { display: false } } }
    });

    // C. Document Heatmap (Minimap)
    renderHeatmap(report.pairs, docLen);
}

function renderHeatmap(pairs, docLen) {
    const container = $('heatmapContainer');
    container.innerHTML = '';
    if (docLen === 0) return;

    // Chỉ vẽ các đoạn High/Very High để đỡ rối
    pairs.filter(p => p.score >= 0.72).forEach(p => {
        const [start, end] = p.a.char_span_in_doc || [0, 0];
        const leftPct = (start / docLen) * 100;
        const widthPct = ((end - start) / docLen) * 100;
        
        const el = document.createElement('div');
        el.className = 'heatmap-marker';
        el.style.left = `${leftPct}%`;
        el.style.width = `${Math.max(0.2, widthPct)}%`; // ít nhất 0.2% để nhìn thấy
        el.title = `Score: ${p.score.toFixed(2)} (Click to view)`;
        el.onclick = () => { jumpToPair(p); };
        
        // Color based on score
        if (p.score >= 0.82) el.style.backgroundColor = 'var(--danger)';
        else el.style.backgroundColor = 'var(--warning)';
        
        container.appendChild(el);
    });
}

// --- 3. Rendering Detailed Report ---
function renderMarkedText(text, spans, level) {
    if (!text) return '';
    // spans: [[start, end], ...] relative to text
    // Sort spans
    spans.sort((a, b) => a[0] - b[0]);
    
    let html = '';
    let cur = 0;
    const lvlClass = `lvl-${level || 'low'}`;
    
    spans.forEach(([s, e]) => {
        s = Math.max(0, s); e = Math.min(text.length, e);
        if (s > cur) html += esc(text.slice(cur, s));
        html += `<mark class="${lvlClass}">${esc(text.slice(s, e))}</mark>`;
        cur = e;
    });
    if (cur < text.length) html += esc(text.slice(cur));
    return html;
}

function renderResultsList() {
    const list = $('resultsList');
    list.innerHTML = '';
    
    if (gFilteredPairs.length === 0) {
        list.innerHTML = '<div class="muted" style="padding:20px; text-align:center">Không tìm thấy kết quả phù hợp bộ lọc.</div>';
        return;
    }

    gFilteredPairs.forEach((p, idx) => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.innerHTML = `
            <div>
                <b>#${idx + 1}</b> 
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${getColorForLevel(p.level)}; margin:0 8px;"></span>
                <span>A[${p.a_chunk_id}] vs B[${p.b_chunk_id}]</span>
            </div>
            <div style="font-weight:bold; color:${getColorForLevel(p.level)}">${(p.score*100).toFixed(1)}%</div>
        `;
        div.onclick = () => selectPair(p, div);
        list.appendChild(div);
    });
}

function getColorForLevel(level) {
    switch(level) {
        case 'very_high': return 'var(--danger)';
        case 'high': return 'var(--warning)';
        case 'medium': return '#eab308'; // darker yellow
        default: return 'var(--text-muted)';
    }
}

function selectPair(pair, el) {
    // Highlight active item in list
    document.querySelectorAll('.result-item').forEach(e => e.classList.remove('active'));
    if (el) el.classList.add('active');

    // Render comparison view
    const showOnlyMatch = $('onlyMatches').checked;
    
    // Helper to get text
    const getHTML = (side) => {
        const text = pair[side].text;
        const spans = pair[side].spans;
        if (showOnlyMatch) {
            // Chỉ hiện các đoạn snippets
            return pair[side].snippets.map(s => 
                `<div style="margin-bottom:10px; border-bottom:1px dashed #eee; padding-bottom:4px;">
                   ...${esc(s.before)}<mark class="lvl-${pair.level}">${esc(s.hit)}</mark>${esc(s.after)}...
                 </div>`
            ).join('');
        } else {
            return `<pre>${renderMarkedText(text, spans, pair.level)}</pre>`;
        }
    };

    $('detailA').innerHTML = getHTML('a');
    $('detailB').innerHTML = getHTML('b');
}

function jumpToPair(pair) {
    // Find index in filtered
    // Đây là logic đơn giản, thực tế cần filter đúng context
    selectPair(pair);
    // Scroll to view logic here...
}

// --- 4. Logic Backend Interaction ---
async function runCompare(e) {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true; btn.innerText = "⏳ Đang chạy...";
    
    const fd = new FormData();
    fd.append('a', $('fileA').files[0]);
    fd.append('b', $('fileB').files[0]);
    fd.append('out', $('outDir').value);
    
    // Fake Progress (UX trick)
    $('progressWrap').classList.remove('hidden');
    let pct = 0;
    const interval = setInterval(() => {
        pct += Math.random() * 5;
        if (pct > 90) pct = 90;
        $('progressBar').style.width = pct + '%';
        $('progressPct').innerText = Math.round(pct) + '%';
    }, 500);

    try {
        const res = await fetch('/api/compare', { method: 'POST', body: fd });
        const data = await res.json();
        
        clearInterval(interval);
        $('progressBar').style.width = '100%';
        $('progressPct').innerText = '100%';
        
        if (data.ok) {
            loadReport(data.report_json);
        } else {
            alert('Lỗi: ' + data.detail);
        }
    } catch (err) {
        alert('Có lỗi xảy ra: ' + err.message);
    } finally {
        btn.disabled = false; btn.innerText = "🚀 Chạy Kiểm Tra";
    }
}

async function loadReport(path) {
    try {
        const res = await fetch(`/api/report?path=${encodeURIComponent(path)}`);
        if(!res.ok){
            const detail = await res.text();
            alert(`Không tải được report (status ${res.status}): ${detail}`);
            return;
        }
        gReport = await res.json();
        
        // Chuyển tab
        document.querySelector('[data-tab="results"]').click();
        
        // Update Metadata
        $('reportMeta').innerText = `Report: ${path.split('/').pop()} | Generated at: ${new Date().toLocaleTimeString()}`;
        
        // Init Data
        gFilteredPairs = gReport.pairs.sort((a,b) => b.score - a.score);
        
        renderCharts(gReport);
        renderResultsList();
        if (gFilteredPairs.length > 0) selectPair(gFilteredPairs[0]);
        
    } catch (e) {
        console.error(e);
        alert('Không tải được report json');
    }
}

// Init Events
$('compare-form').onsubmit = runCompare;
$('filterLevel').onchange = () => {
    const val = $('filterLevel').value;
    gFilteredPairs = (gReport?.pairs||[]).filter(p => !val || p.level === val);
    renderResultsList();
};
$('onlyMatches').onchange = () => {
    // Re-render current pair selection
    const active = document.querySelector('.result-item.active');
    if (active) active.click();
};
