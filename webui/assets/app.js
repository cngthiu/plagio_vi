// ================================
// file: webui/assets/app.js  (FULL)
// ================================
let gReport=null, gReportPath=null, gPairsFiltered=[], gPage=1, gPageSize=50;
let donut=null, progressTimer=null, progressVal=0;
let gSelectedIndex=-1, gActiveCard=null;

const $ = (id)=>document.getElementById(id);
const esc = (s)=> (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const runStatusEl = $('runStatus');
const emptyResultsEl = $('emptyResults');
const btnRun = $('btnRun');
const pageSizeSelect = $('pageSize');
if(pageSizeSelect){
  gPageSize = parseInt(pageSizeSelect.value || '50', 10);
}

function setTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.tab===name));
  document.querySelectorAll('.tabpane').forEach(p=>p.classList.toggle('active', p.id===`tab-${name}`));
}
$('btnDark').onclick = ()=>{
  const html=document.documentElement;
  html.setAttribute('data-theme', html.getAttribute('data-theme')==='dark'?'light':'dark');
};

function setStatus(text, tone='subtle'){
  if(!runStatusEl) return;
  runStatusEl.textContent = text;
  runStatusEl.className = `status-pill ${tone}`;
}

// ---------- donuts / metrics ----------
function renderDonut(percent){
  percent = Math.max(0, Math.min(100, percent));
  const ctx=$('donutScore').getContext('2d');
  if(donut) donut.destroy();
  donut = new Chart(ctx, {
    type:'doughnut',
    data:{ datasets:[{ data:[percent, 100-percent] }]},
    options:{ cutout:'65%', plugins:{legend:{display:false}}, animation:{duration:400} }
  });
  $('pctText').innerText=`${percent.toFixed(0)}%`;
}

function unionIntervals(intervals){
  if(!intervals.length) return [];
  intervals.sort((a,b)=>a[0]-b[0]);
  const out=[intervals[0].slice()];
  for(const [s,e] of intervals.slice(1)){
    const last=out[out.length-1];
    if(s <= last[1]) last[1]=Math.max(last[1], e);
    else out.push([s,e]);
  }
  return out;
}
function countWordsByChars(chars){ return Math.round(chars/5.5); }
function renderMarked(text, spans){
  const content = text || '';
  const markers = (spans || []).slice().sort((a,b)=>a[0]-b[0]);
  let cur = 0;
  const buf = ['<pre>'];
  for(const [rawS, rawE] of markers){
    const s = Math.max(0, Math.min(content.length, rawS));
    const e = Math.max(0, Math.min(content.length, rawE));
    if(cur < s) buf.push(esc(content.slice(cur, s)));
    if(e > s) buf.push('<mark>'+esc(content.slice(s, e))+'</mark>');
    cur = Math.max(cur, e);
  }
  if(cur < content.length){
    buf.push(esc(content.slice(cur)));
  }
  buf.push('</pre>');
  return `<div class="result-text">${buf.join('')}</div>`;
}
function buildDocMetrics(report){
  let docLen = 0;
  const globalSpansA=[];
  for(const r of (report.pairs||[])){
    const baseA = (r.a?.char_span_in_doc||[0,0])[0];
    const spans = r.a?.spans || [];
    for(const [s,e] of spans){
      const g0=baseA + s, g1=baseA + e;
      if(g1>g0) globalSpansA.push([g0,g1]);
      if(r.a?.char_span_in_doc?.[1]>docLen) docLen = r.a.char_span_in_doc[1];
    }
  }
  const merged = unionIntervals(globalSpansA);
  const matchedChars = merged.reduce((acc,[s,e])=>acc+(e-s),0);
  const percent = docLen>0 ? (matchedChars/docLen*100):0;
  const matchedWords = countWordsByChars(matchedChars);
  return { docLen, matchedChars, percent, matchedWords };
}

// ---------- Results list ----------
function renderResultsPage(){
  const list=$('resultsList');
  if(!list){ return; }
  list.innerHTML='';
  const totalPages = Math.max(1, Math.ceil(Math.max(1, gPairsFiltered.length)/gPageSize));
  gPage = Math.min(Math.max(1, gPage), totalPages);
  const start=(gPage-1)*gPageSize;
  const end=Math.min(start+gPageSize, gPairsFiltered.length);
  if(gPairsFiltered.length){
    if(gSelectedIndex < start || gSelectedIndex >= end){
      gSelectedIndex = start;
    }
  } else {
    gSelectedIndex = -1;
  }
  const detailRecord = (gSelectedIndex >=0 && gSelectedIndex < gPairsFiltered.length) ? gPairsFiltered[gSelectedIndex] : null;
  const pageInfo=$('pageInfo');
  if(pageInfo){
    pageInfo.innerText = gPairsFiltered.length ? `Trang ${gPage}/${Math.max(1, Math.ceil(gPairsFiltered.length/gPageSize))}` : '';
  }
  gActiveCard = null;
  for(let offset=0; start + offset < end; offset++){
    const idx = start + offset;
    const r = gPairsFiltered[idx];
    const matchChars = (r.a?.spans||[]).reduce((acc,[s,e])=>acc + Math.max(0, e - s), 0);
    const matchWords = countWordsByChars(matchChars);
    const matchCount = (r.a?.spans || []).length;
    const card=document.createElement('div');
    card.className='result-card';
    if(idx === gSelectedIndex){ card.classList.add('active'); gActiveCard = card; }
    const firstSnippet = (r.a?.snippets||[])[0]?.hit || '';
    card.innerHTML = `
      <div class="result-header">
        <span class="result-index">Cặp ${idx+1}</span>
        <div class="result-meta">
          <span class="result-pill">~${matchWords} từ trùng</span>
          <span class="result-pill">${matchCount} cụm</span>
        </div>
      </div>
      <div class="result-body">
        <div class="result-section">
          <div class="result-label">Tài liệu A</div>
          ${renderMarked(r.a?.text||'', r.a?.spans||[])}
        </div>
        <div class="result-section">
          <div class="result-label">Nguồn khớp</div>
          ${renderMarked(r.b?.text||'', r.b?.spans||[])}
        </div>
      </div>
      ${firstSnippet ? `<div class="result-snippet">Cụm nổi bật: <mark>${esc(firstSnippet)}</mark></div>` : ''}`;
    card.onclick=()=>{
      gSelectedIndex = idx;
      if(gActiveCard){ gActiveCard.classList.remove('active'); }
      card.classList.add('active');
      gActiveCard = card;
      showDetail(r);
    };
    list.appendChild(card);
  }
  if(detailRecord){
    showDetail(detailRecord);
  } else {
    const detailA=$('detailA'); if(detailA) detailA.innerHTML='';
    const detailB=$('detailB'); if(detailB) detailB.innerHTML='';
    $('snipsA').innerHTML=''; $('snipsB').innerHTML='';
    $('matchListA').innerHTML=''; $('matchListB').innerHTML='';
  }
}
function updateEmptyState(){
  if(!emptyResultsEl) return;
  const has = gPairsFiltered.length > 0;
  emptyResultsEl.classList.toggle('hidden', has);
  const list = $('resultsList');
  if(list){
    list.classList.toggle('hidden', !has);
  }
}
function applyFilters(){
  if(!gReport || !Array.isArray(gReport.pairs)) {
    gPairsFiltered = [];
    const list=$('resultsList'); if(list) list.innerHTML='';
    updateEmptyState();
    return;
  }
  const level=$('filterLevel').value, q=$('searchText').value.trim().toLowerCase();
  const sortBy=$('sortBy').value;
  let pairs=gReport.pairs.slice();
  if(level) pairs=pairs.filter(r=>r.level===level);
  if(q){ pairs=pairs.filter(r=>(r.a?.text||'').toLowerCase().includes(q) || (r.b?.text||'').toLowerCase().includes(q)); }
  if(sortBy === 'score_desc'){
    pairs.sort((a,b)=> (b.score??0) - (a.score??0));
  } else if(sortBy === 'a_order'){
    pairs.sort((a,b)=> (a.a_chunk_id??0) - (b.a_chunk_id??0));
  }
  gPairsFiltered=pairs;
  gPage=1;
  gSelectedIndex = gPairsFiltered.length ? 0 : -1;
  renderResultsPage();
  updateEmptyState();
}

// ---------- Sources ----------
function renderSources(report){
  const box=$('sourcesList'); box.innerHTML='';
  const map=new Map();
  for(const r of (report.pairs||[])){
    const spans=r.b?.spans||[];
    let chars=0; for(const [s,e] of spans) chars += Math.max(0, e-s);
    const words = Math.max(1, countWordsByChars(chars));
    const item = map.get(r.b_chunk_id) || {words:0, score:0, text:r.b?.text||'', id:r.b_chunk_id, hits:0};
    item.words += words; item.score = Math.max(item.score, r.score||0); item.hits += spans.length;
    map.set(r.b_chunk_id, item);
  }
  const arr=[...map.values()].sort((a,b)=> b.words-a.words).slice(0,8);
  if(!arr.length){ box.innerHTML='<div class="muted">Chưa có nguồn đáng chú ý.</div>'; return; }
  for(const s of arr){
    const div=document.createElement('div'); div.className='source';
    div.innerHTML=`
      <div><b>B[${s.id}]</b> <span class="tag">~${s.words} từ trùng</span> <span class="tag">Max score ${s.score.toFixed(3)}</span></div>
      <div class="preview">${esc(s.text.slice(0,240))}</div>`;
    box.appendChild(div);
  }
  $('topSource').innerText = arr.length ? `B[${arr[0].id}] ~${arr[0].words} từ` : '—';
}

// ---------- Overview ----------
function renderOverview(report){
  const met = buildDocMetrics(report);
  $('statPairs').innerText = report.summary?.pairs ?? report.pairs.length ?? 0;
  $('statChunksA').innerText = report.summary?.chunks_A ?? 0;
  $('statChunksB').innerText = report.summary?.chunks_B ?? 0;
  $('coverText').innerText = `${met.matchedChars} / ${met.docLen}`;
  $('wordsText').innerText = `${met.matchedWords} từ trùng`;
  renderDonut(met.percent);

  const top = (report.pairs||[]).slice().sort((a,b)=> (b.score||0)-(a.score||0)).slice(0,5);
  const box=$('docAOverview'); box.innerHTML='';
  for(const r of top){
    const card=document.createElement('div'); card.className='source';
    card.innerHTML = `<div class="muted small">A[${r.a_chunk_id}] vs B[${r.b_chunk_id}] — score ${r.score?.toFixed(3)}</div>` +
      renderMarked(r.a?.text||'', r.a?.spans||[]);
    box.appendChild(card);
  }
  renderSources(report);
}

// ---------- Detail (two-column + match pairs) ----------
function showDetail(r){
  const only = $('onlyMatches').checked;
  const aText = r.a?.text || '', bText = r.b?.text || '';
  const aSpans = r.a?.spans || [], bSpans = r.b?.spans || [];

  $('detailA').innerHTML = only ? renderOnlyMatches(aText, aSpans) : renderMarked(aText, aSpans);
  $('detailB').innerHTML = only ? renderOnlyMatches(bText, bSpans) : renderMarked(bText, bSpans);

  $('snipsA').innerHTML = (r.a?.snippets||[]).slice(0,3).map(sn=>`<pre>...${esc(sn.before)}<mark>${esc(sn.hit)}</mark>${esc(sn.after)}...</pre>`).join('');
  $('snipsB').innerHTML = (r.b?.snippets||[]).slice(0,3).map(sn=>`<pre>...${esc(sn.before)}<mark>${esc(sn.hit)}</mark>${esc(sn.after)}...</pre>`).join('');

  // Cụm trùng A↔B: zip theo thứ tự (runner giữ thứ tự cặp sau NMS)
  const pairsA = aSpans.slice();
  const pairsB = bSpans.slice();
  const n = Math.min(pairsA.length, pairsB.length);
  const listA = $('matchListA'); const listB = $('matchListB');
  listA.innerHTML = ''; listB.innerHTML = '';
  for(let i=0;i<n;i++){
    const [sa,ea]=pairsA[i], [sb,eb]=pairsB[i];
    const divA=document.createElement('div'); divA.className='matchitem';
    divA.innerHTML = esc(aText.slice(sa,ea));
    const divB=document.createElement('div'); divB.className='matchitem';
    divB.innerHTML = esc(bText.slice(sb,eb));
    listA.appendChild(divA); listB.appendChild(divB);
  }
}
function renderOnlyMatches(text, spans){
  if(!spans?.length) return `<pre>${esc(text||'')}</pre>`;
  const buf = spans.slice().sort((a,b)=>a[0]-b[0]).map(([s,e]) => text.slice(Math.max(0,s), Math.max(0,Math.min(text.length,e))));
  return `<pre>${buf.map(t=>`<mark>${esc(t)}</mark>`).join(' ... ')}</pre>`;
}
$('onlyMatches').addEventListener('change', ()=>{
  const row = gPairsFiltered[(gPage-1)*gPageSize]; // render lại từ bản đầu trang
  if(row) showDetail(row);
});

// Synchronized scroll
(function syncScroll(){
  const panes = document.querySelectorAll('.syncpane');
  let lock=false;
  panes.forEach(p=>{
    p.addEventListener('scroll', ()=>{
      if(lock) return;
      lock=true;
      panes.forEach(q=>{
        if(q===p) return;
        q.scrollTop = (p.scrollTop / (p.scrollHeight - p.clientHeight)) * (q.scrollHeight - q.clientHeight);
      });
      lock=false;
    });
  });
})();

// ---------- Load report ----------
async function loadReport(path){
  setStatus('Đang mở report...', 'loading');
  const res = await fetch(`/api/report?path=${encodeURIComponent(path)}`);
  if(!res.ok){ alert('Không mở được report.json'); setStatus('Không mở được report', 'error'); return; }
  let data = await res.json();
  if(typeof data==='string'){ try{ data=JSON.parse(data); }catch{ alert('Report không hợp lệ'); setStatus('Report không hợp lệ', 'error'); return; } }
  if(!data || !Array.isArray(data.pairs)){ alert('Report thiếu "pairs"'); setStatus('Report thiếu dữ liệu', 'error'); return; }
  gReport = data; gReportPath = path;
  setTab('results');
  renderOverview(gReport);
  applyFilters();
  $('btnOpenHtml').onclick=(e)=>{ e.preventDefault(); window.open(`/api/download_report_html?path=${encodeURIComponent(path.replace(/report\.json$/,'report.html'))}`,'_blank'); };
  setStatus('Đã mở report có sẵn', 'ok');
}

// ---------- Compare (progress ramp) ----------
function startProgress(){
  $('progressWrap').classList.remove('hidden');
  progressVal=0; $('progressBar').style.width='0%'; $('progressPct').innerText='0%';
  clearInterval(progressTimer);
  progressTimer = setInterval(()=>{
    if(progressVal < 95){
      progressVal += Math.max(0.2, (95-progressVal)*0.03);
      $('progressBar').style.width=`${progressVal.toFixed(0)}%`;
      $('progressPct').innerText=`${progressVal.toFixed(0)}%`;
    }
  }, 300);
}
function finishProgress(){
  clearInterval(progressTimer);
  progressVal=100;
  $('progressBar').style.width='100%'; $('progressPct').innerText='100%';
  setTimeout(()=> $('progressWrap').classList.add('hidden'), 800);
}
function readSettingsUI(){
  return {
    weights: {
      w_cross: parseFloat($('w_cross').value||'0.55'),
      w_bi: parseFloat($('w_bi').value||'0.25'),
      w_lex: parseFloat($('w_lex').value||'0.10'),
      w_bm25: parseFloat($('w_bm25').value||'0.10'),
    },
    thresholds: {
      very_high: parseFloat($('th_vh').value||'0.82'),
      high: parseFloat($('th_h').value||'0.72'),
      medium: parseFloat($('th_m').value||'0.60'),
    },
    penalties: {
      lex_boilerplate_lambda: parseFloat($('pen_lambda').value||'0.5'),
      min_span_chars: parseInt($('min_span').value||'24'),
      small_span_chars: parseInt($('small_span').value||'12'),
      min_small_spans: parseInt($('min_small').value||'2'),
    },
    retrieval: {
      bm25_topk: parseInt($('bm25_topk').value||'30'),
      ann_topk_recall: parseInt($('ann_topk').value||'50'),
      rerank_topk: parseInt($('rerank_topk').value||'8'),
      simhash_topk: parseInt($('simhash_topk').value||'40'),
    },
    chunking: {
      size_tokens: parseInt($('size_tokens').value||'160'),
      overlap: parseInt($('overlap').value||'40'),
      max_seq_len_bi: parseInt($('max_seq_bi').value||'256'),
      max_seq_len_cross: parseInt($('max_seq_cross').value||'384'),
    }
  };
}
function fillSettingsUI(obj){
  if(!obj) return;
  const w=obj.weights||{}, th=obj.thresholds||{}, p=obj.penalties||{}, r=obj.retrieval||{}, c=obj.chunking||{};
  $('w_cross').value=w.w_cross??0.55; $('w_bi').value=w.w_bi??0.25; $('w_lex').value=w.w_lex??0.10; $('w_bm25').value=w.w_bm25??0.10;
  $('th_vh').value=th.very_high??0.82; $('th_h').value=th.high??0.72; $('th_m').value=th.medium??0.60;
  $('pen_lambda').value=p.lex_boilerplate_lambda??0.5; $('min_span').value=p.min_span_chars??24; $('small_span').value=p.small_span_chars??12; $('min_small').value=p.min_small_spans??2;
  $('bm25_topk').value=r.bm25_topk??30; $('ann_topk').value=r.ann_topk_recall??50; $('rerank_topk').value=r.rerank_topk??8; $('simhash_topk').value=r.simhash_topk??40;
  $('size_tokens').value=c.size_tokens??160; $('overlap').value=c.overlap??40; $('max_seq_bi').value=c.max_seq_len_bi??256; $('max_seq_cross').value=c.max_seq_len_cross??384;
}
$('btnLoadSettings').onclick = async ()=>{
  const res = await fetch(`/api/settings`);
  const data = await res.json();
  const saved = JSON.parse(localStorage.getItem('plagio_settings')||'null');
  fillSettingsUI(saved || data);
};
$('btnSaveSettings').onclick = ()=>{
  localStorage.setItem('plagio_settings', JSON.stringify(readSettingsUI()));
  alert('Đã lưu settings vào trình duyệt.');
};

async function runCompare(){
  const aFile=$('fileA').files[0], bFile=$('fileB').files[0];
  if(!aFile || !bFile){ alert('Chọn đủ 2 file DOCX'); return; }
  setStatus('Đang so khớp...', 'loading');
  if(btnRun){ btnRun.disabled = true; btnRun.setAttribute('aria-busy','true'); }
  startProgress();
  const fd=new FormData();
  fd.append('a', aFile); fd.append('b', bFile);
  fd.append('out', $('outDir').value.trim() || 'outputs/webui_run');
  const settings = localStorage.getItem('plagio_settings') || JSON.stringify(readSettingsUI());
  fd.append('settings', settings);
  try{
    const res=await fetch('/api/compare', {method:'POST', body:fd});
    const data=await res.json();
    if(!res.ok || !data.ok) throw new Error(data.detail||'Compare failed');
    setStatus('Hoàn tất ✓ — đang tải báo cáo...', 'loading');
    $('reportPath').value=data.report_json;
    await loadReport(data.report_json);
    setStatus('Hoàn tất ✓', 'ok');
    finishProgress();
  }catch(e){
    console.error(e);
    setStatus(`Lỗi: ${e.message}`, 'error');
    finishProgress();
    alert('Lỗi so khớp: '+e.message);
  } finally{
    if(btnRun){ btnRun.disabled = false; btnRun.removeAttribute('aria-busy'); }
  }
}

// ---------- Events ----------
document.querySelectorAll('.tab').forEach(btn=> btn.addEventListener('click', ()=> setTab(btn.dataset.tab)));
const compareForm = $('compare-form');
if(compareForm){
  compareForm.addEventListener('submit', (e)=>{ e.preventDefault(); runCompare(); });
}
const btnLoadReport = $('btnLoadReport');
if(btnLoadReport){
  btnLoadReport.addEventListener('click', ()=>{
    const p=$('reportPath').value.trim();
    if(!p){ alert('Nhập report.json'); return; }
    loadReport(p);
  });
}
$('filterLevel').addEventListener('change', applyFilters);
$('sortBy').addEventListener('change', applyFilters);
$('searchText').addEventListener('input', applyFilters);
const prevBtn = $('prevPage');
if(prevBtn){
  prevBtn.addEventListener('click', ()=>{
    if(gPage>1){
      gPage--;
      renderResultsPage();
    }
  });
}
const nextBtn = $('nextPage');
if(nextBtn){
  nextBtn.addEventListener('click', ()=>{
    const max=Math.max(1, Math.ceil(gPairsFiltered.length/gPageSize));
    if(gPage<max){
      gPage++;
      renderResultsPage();
    }
  });
}
if(pageSizeSelect){
  pageSizeSelect.addEventListener('change', ()=>{
    gPageSize = parseInt(pageSizeSelect.value || '50', 10);
    gPage = 1;
    renderResultsPage();
    updateEmptyState();
  });
}
const btnExport = $('btnExportCSV');
if(btnExport){
  btnExport.addEventListener('click', ()=>{
    if(!gReportPath){
      alert('Chưa mở report');
      return;
    }
    window.open(`/api/export_csv?report_path=${encodeURIComponent(gReportPath)}`,'_blank');
  });
}
updateEmptyState();
setStatus('Sẵn sàng', 'subtle');
