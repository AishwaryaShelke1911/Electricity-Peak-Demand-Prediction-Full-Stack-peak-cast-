const API = ""; // same-origin — Flask serves this file too
const MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"];

function kToF(k){ return Math.round((k - 273.15) * 9/5 + 32); }

/* ======================= CALENDAR (real AEP data) ======================= */

let calState = { year: 2013, month: 7 }; // sensible default: a peak summer month
let dateBounds = { min: "2012-10-01", max: "2017-11-30" };
let selectedCell = null;

async function initCalendar(){
  try{
    const res = await fetch(`${API}/api/history/date-range`);
    dateBounds = await res.json();
    document.getElementById('apiStatus').textContent = 'live from Flask API';
  }catch(e){
    document.getElementById('apiStatus').textContent = 'backend not reachable — is app.py running?';
  }
  renderCalendar();
}

function boundsAsYM(){
  const [minY, minM] = dateBounds.min.split('-').map(Number);
  const [maxY, maxM] = dateBounds.max.split('-').map(Number);
  return {minY, minM, maxY, maxM};
}

async function renderCalendar(){
  const {year, month} = calState;
  document.getElementById('calLabel').textContent = `${MONTH_NAMES[month-1]} ${year}`;
  const {minY, minM, maxY, maxM} = boundsAsYM();
  document.getElementById('prevMonth').disabled = (year===minY && month===minM);
  document.getElementById('nextMonth').disabled = (year===maxY && month===maxM);

  const grid = document.getElementById('calGrid');
  grid.innerHTML = '<p class="small">Loading…</p>';

  let days = [];
  try{
    const res = await fetch(`${API}/api/history/calendar/${year}/${month}`);
    days = await res.json();
  }catch(e){
    grid.innerHTML = '<p class="small">Could not load this month.</p>';
    return;
  }

  const byDate = {};
  days.forEach(d => byDate[d.date] = d);
  const firstWeekday = new Date(year, month-1, 1).getDay();
  const daysInMonth = new Date(year, month, 0).getDate();

  grid.innerHTML = '';
  for(let i=0;i<firstWeekday;i++){
    const cell = document.createElement('div');
    cell.className = 'cal-cell empty';
    grid.appendChild(cell);
  }
  for(let d=1; d<=daysInMonth; d++){
    const dateStr = `${year}-${String(month).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
    const info = byDate[dateStr];
    const cell = document.createElement('div');
    cell.textContent = d;
    cell.className = 'cal-cell' + (info ? ' risk-' + info.risk.toLowerCase() : ' risk-normal');
    cell.title = info ? `${dateStr} — ${(info.prob*100).toFixed(1)}% peak-day probability` : dateStr;
    cell.addEventListener('click', () => selectDate(dateStr, cell));
    grid.appendChild(cell);
  }
}

document.getElementById('prevMonth').addEventListener('click', () => {
  calState.month--; if(calState.month < 1){ calState.month = 12; calState.year--; }
  renderCalendar();
});
document.getElementById('nextMonth').addEventListener('click', () => {
  calState.month++; if(calState.month > 12){ calState.month = 1; calState.year++; }
  renderCalendar();
});

async function selectDate(dateStr, cellEl){
  if(selectedCell) selectedCell.classList.remove('selected');
  if(cellEl){ cellEl.classList.add('selected'); selectedCell = cellEl; }

  document.getElementById('dayHint').style.display = 'none';
  document.getElementById('dayContent').style.display = 'block';

  const res = await fetch(`${API}/api/history/day/${dateStr}`);
  if(!res.ok){
    document.getElementById('dayHint').style.display = 'block';
    document.getElementById('dayHint').textContent = 'No data for that date.';
    document.getElementById('dayContent').style.display = 'none';
    return;
  }
  const d = await res.json();

  document.getElementById('dayDate').textContent = d.date + (d.is_weekend ? ' (weekend)' : ' (weekday)');
  const badge = document.getElementById('dayRiskBadge');
  badge.textContent = d.risk + ' risk';
  badge.className = 'badge ' + d.risk;

  document.getElementById('dayProb').textContent = (d.peak_day_prob*100).toFixed(2) + '%';
  document.getElementById('dayMaxLoad').textContent = Math.round(d.daily_max_load).toLocaleString();
  document.getElementById('dayTemp').textContent = kToF(d.max_temperature_k) + '°F';
}

/* ======================= RESULTS (real published metrics) ======================= */

async function loadResults(){
  const res = await fetch(`${API}/api/history/metrics`);
  const m = await res.json();

  const dayCards = document.querySelectorAll('#dayMetricsCards .card');
  dayCards[0].querySelector('.n').textContent = (m.peak_day_model.accuracy*100).toFixed(1)+'%';
  dayCards[1].querySelector('.n').textContent = m.peak_day_model.roc_auc;
  dayCards[2].querySelector('.n').textContent = m.peak_day_model.f1;

  const hourCards = document.querySelectorAll('#hourMetricsCards .card');
  hourCards[0].querySelector('.n').textContent = (m.peak_hour_model.accuracy*100).toFixed(1)+'%';
  hourCards[1].querySelector('.n').textContent = m.peak_hour_model.roc_auc;
  hourCards[2].querySelector('.n').textContent = m.peak_hour_model.max_prob;
}

/* ======================= INIT ======================= */

initCalendar();
loadResults();