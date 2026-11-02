import { el, $, clear, t, LANGS, getLang, setLang, initTheme, toggleTheme, makeApi, connectSocket, fmtDateTime } from '/shared/common.js';

initTheme();
const app = $('#app');
let token = null, user = null;
try { token = sessionStorage.getItem('atok'); user = JSON.parse(sessionStorage.getItem('auser') || 'null'); } catch { /* ignore */ }
const api = makeApi(() => token, () => logout());
const toast = (m) => { const n = el('div', { class: 'toast' }, m); $('#toasts').append(n); setTimeout(() => n.remove(), 5000); };
const TABS = ['overview', 'desks', 'employees', 'citizens', 'signs', 'training', 'settings', 'logs'];
const TITLES = { overview: 'Overview', desks: 'Desks & cameras', employees: 'Employees', citizens: 'Citizens & Face ID', signs: 'Signs & dataset',
  training: 'Training & models', settings: 'Settings', logs: 'Logs' };
const S = { tab: 'overview', err: '', sock: null, trainTimer: null };

function save() { try { sessionStorage.setItem('atok', token || ''); sessionStorage.setItem('auser', JSON.stringify(user)); } catch { /* ignore */ } }
async function login(u, p) {
  try {
    const r = await api('/api/auth/login', { method: 'POST', body: { username: u, password: p } });
    if (r.user.role !== 'ADMIN') { S.err = 'Administrator account required'; return render(); }
    token = r.access_token; user = r.user; save(); S.err = ''; openSocket(); render();
  } catch (e) { S.err = e.message; render(); }
}
function logout() { token = null; user = null; save(); S.sock && S.sock.close(); S.sock = null; clearInterval(S.trainTimer); render(); }
setInterval(async () => { if (token) { try { const r = await api('/api/auth/refresh', { method: 'POST' }); token = r.access_token; save(); } catch { /* ignore */ } } }, 4 * 60 * 1000);

function openSocket() {
  S.sock = connectSocket({ getToken: () => token, onEvent: (ev) => {
    if (ev.type === 'enroll.done') toast(ev.ok ? '✓ Face template saved for ' + (ev.citizen_code || '') : '✗ Enrolment failed: ' + ev.error);
    if (ev.type === 'record.saved') { toast('✓ Sample saved for ' + ev.sign + ' (' + ev.frames + ' frames)'); if (S.tab === 'signs') show(); }
    if (ev.type === 'record.failed') toast('✗ Recording failed: ' + ev.error);
    if (ev.type === 'assist.requested') toast('🙋 Assistance requested at ' + ev.desk_code);
  }, onState: () => {} });
}

// ------------------------------------------------------------------ small UI helpers
const table = (cols, rows) => el('table', {}, el('thead', {}, el('tr', {}, cols.map((c) => el('th', {}, c)))),
  el('tbody', {}, rows.length ? rows.map((r) => el('tr', {}, r.map((c) => el('td', {}, c)))) : el('tr', {}, el('td', { colspan: cols.length, class: 'muted' }, '—'))));
const field = (label, input) => el('div', { class: 'field' }, el('label', {}, label), input);
const input = (type = 'text', ph = '', val = '') => el('input', { type, placeholder: ph, value: val });
const pct = (x) => (x == null ? '—' : (x * 100).toFixed(1) + '%');
const card = (title, ...kids) => el('div', { class: 'card' }, title ? el('h2', {}, title) : null, ...kids);
async function act(fn, ok) { try { const r = await fn(); if (ok) toast(ok); return r; } catch (e) { toast('✗ ' + e.message); } }

// ------------------------------------------------------------------ tabs
async function overview() {
  const s = await api('/api/admin/stats');
  const max = Math.max(1, ...s.per_sign.map((x) => x.count));
  return [
    s.model && s.model.synthetic ? el('div', { class: 'banner warn' }, '⚠ The active model was trained on SYNTHETIC data. It is a pipeline test, not Uzbek Sign Language. Record real samples and retrain.') : null,
    !s.model ? el('div', { class: 'banner bad' }, 'No model is active. Open “Training & models”.') : null,
    el('div', { class: 'grid' }, [['Active sessions', s.active_sessions], ['Sessions (24 h)', s.sessions_24h], ['Citizens', s.citizens], ['Employees', s.employees],
      ['AI services online', s.ai_services_online], ['Recognitions (7 d)', s.recognitions_7d], ['Accept rate (7 d)', pct(s.accept_rate_7d)], ['Avg confidence', pct(s.avg_confidence_7d)]]
      .map(([k, v]) => el('div', { class: 'card stat' }, el('h2', {}, k), el('div', { class: 'n' }, String(v))))),
    card('Most recognised signs (7 days)', s.per_sign.length ? s.per_sign.map((x) => el('div', { class: 'bar-chart' }, el('span', { style: 'width:150px' }, x.sign),
      el('span', { class: 'b', style: `width:${(x.count / max) * 60}%` }), el('span', { class: 'muted' }, `${x.count} · ${pct(x.avg_confidence)}`))) : el('div', { class: 'muted' }, '—')),
    card('Active model', s.model ? table(['Version', 'Arch', 'Classes', 'Val acc', 'Test acc', 'Data'], [[s.model.version, s.model.arch, s.model.num_classes, pct(s.model.val_accuracy), pct(s.model.test_accuracy), s.model.synthetic ? 'SYNTHETIC' : 'real']]) : '—'),
  ];
}

async function desks() {
  const list = await api('/api/admin/desks');
  const code = input('text', 'DESK-2'), name = input('text', 'Service desk 2');
  return [card('Service desks & cameras', table(['Code', 'Name', 'Camera #', 'Camera permission', 'AI service', 'Active'], list.map((d) => {
    const idx = el('input', { type: 'number', min: 0, max: 16, value: d.camera_index, style: 'width:80px', onchange: (e) => act(() => api('/api/admin/desks/' + d.id, { method: 'PATCH', body: { camera_index: Number(e.target.value) } }), 'Camera index saved') });
    return [d.code, d.name, idx,
      el('button', { class: 'btn small ' + (d.camera_enabled ? '' : 'danger'), onclick: async () => { await act(() => api('/api/admin/desks/' + d.id, { method: 'PATCH', body: { camera_enabled: !d.camera_enabled } })); show(); } }, d.camera_enabled ? 'Allowed' : 'Blocked'),
      el('span', { class: 'pill ' + (d.ai_online ? 'ok' : 'bad') }, d.ai_online ? 'online' : 'offline'), d.active ? 'yes' : 'no'];
  }))),
  card('Add desk', el('div', { class: 'inline-form' }, field('Code', code), field('Name', name),
    el('button', { class: 'btn primary', onclick: async () => { await act(() => api('/api/admin/desks', { method: 'POST', body: { code: code.value, name: name.value } }), 'Desk added'); show(); } }, 'Add')))];
}

async function employees() {
  const [list, ds] = await Promise.all([api('/api/admin/employees'), api('/api/admin/desks')]);
  const f = { username: input(), full: input(), pw: input('password'), desk: el('select', {}, el('option', { value: '' }, '—'), ds.map((d) => el('option', { value: d.id }, d.code))),
    role: el('select', {}, el('option', { value: 'EMPLOYEE' }, 'EMPLOYEE'), el('option', { value: 'ADMIN' }, 'ADMIN')) };
  return [card('Employees', table(['Username', 'Name', 'Role', 'Desk', 'Status', ''], list.map((u) => [u.username, u.full_name, u.role, (ds.find((d) => d.id === u.desk_id) || {}).code || '—',
    u.active ? 'active' : 'removed', u.active ? el('button', { class: 'btn small danger', onclick: async () => { if (confirm('Remove ' + u.username + '?')) { await act(() => api('/api/admin/employees/' + u.id, { method: 'DELETE' }), 'Removed'); show(); } } }, 'Remove') : '']))),
  card('Add employee', el('div', { class: 'inline-form' }, field('Username', f.username), field('Full name', f.full), field('Password (min 8)', f.pw), field('Desk', f.desk), field('Role', f.role),
    el('button', { class: 'btn primary', onclick: async () => { await act(() => api('/api/admin/employees', { method: 'POST', body: { username: f.username.value, full_name: f.full.value, password: f.pw.value, desk_id: f.desk.value ? Number(f.desk.value) : null, role: f.role.value } }), 'Employee added'); show(); } }, 'Add')))];
}

async function citizens() {
  const [list, ds] = await Promise.all([api('/api/admin/citizens'), api('/api/admin/desks')]);
  const online = ds.filter((d) => d.ai_online);
  const deskSel = el('select', { style: 'width:auto' }, (online.length ? online : ds).map((d) => el('option', { value: d.id }, d.code + (d.ai_online ? '' : ' (AI offline)'))));
  const f = { name: input(), nid: input('text', 'optional'), pin: input('password', '4–8 digits'), script: el('select', {}, el('option', { value: 'latin' }, 'Latin'), el('option', { value: 'cyrillic' }, 'Cyrillic')) };
  return [card('Register citizen', el('div', { class: 'inline-form' }, field('Full name', f.name), field('National ID (stored only as a keyed hash)', f.nid), field('Fallback PIN', f.pin), field('Script', f.script),
    el('button', { class: 'btn primary', onclick: async () => { await act(() => api('/api/admin/citizens', { method: 'POST', body: { full_name: f.name.value, national_id: f.nid.value || null, pin: f.pin.value || null, script: f.script.value } }), 'Citizen registered'); show(); } }, 'Register'))),
  card('Citizens', el('div', { class: 'row', style: 'margin-bottom:.8rem' }, el('span', { class: 'muted' }, 'Enrol Face ID at desk:'), deskSel),
    table(['Code', 'Name', 'PIN', 'Face templates', ''], list.map((c) => [c.code, c.full_name, c.has_pin ? '✓' : '—', c.face_templates,
      el('div', { class: 'row' },
        el('button', { class: 'btn small primary', onclick: () => act(() => api(`/api/admin/citizens/${c.id}/enroll-face`, { method: 'POST', body: { desk_id: Number(deskSel.value) } }), 'Ask the citizen to look at the camera, blink and turn their head slowly…') }, 'Register Face ID'),
        c.face_templates ? el('button', { class: 'btn small', onclick: async () => { if (confirm('Delete face templates of ' + c.code + '?')) { await act(() => api(`/api/admin/citizens/${c.id}/face`, { method: 'DELETE' }), 'Face data deleted'); show(); } } }, 'Delete face') : null,
        el('button', { class: 'btn small danger', onclick: async () => { if (confirm('Delete ' + c.code + ' and all biometric data?')) { await act(() => api('/api/admin/citizens/' + c.id, { method: 'DELETE' }), 'Citizen deleted'); show(); } } }, 'Delete')),
    ])))];
}

async function signs() {
  const [list, ds, stats] = await Promise.all([api('/api/admin/signs'), api('/api/admin/desks'), api('/api/admin/dataset')]);
  const online = ds.filter((d) => d.ai_online);
  const deskSel = el('select', { style: 'width:auto' }, (online.length ? online : ds).map((d) => el('option', { value: d.id }, d.code + (d.ai_online ? '' : ' (AI offline)'))));
  const speaker = input('text', 'speaker id (e.g. ali)'), dur = el('input', { type: 'number', min: 1, max: 10, step: 0.5, value: 3, style: 'width:90px' });
  const f = { name: input('text', 'PASPORT'), latin: input('text', 'pasport'), cyr: input('text', 'optional'), cat: input('text', 'document'),
    pos: el('select', {}, ['noun', 'verb', 'pron', 'adj', 'interj', 'adv'].map((p) => el('option', {}, p))) };
  return [card('Record training samples', el('div', { class: 'row' }, el('span', { class: 'muted' }, 'Desk camera:'), deskSel, speaker, el('span', { class: 'muted' }, 'seconds:'), dur),
    el('div', { class: 'muted', style: 'margin:.5rem 0' }, 'Choose a sign below. The desk shows nothing special: after a 2-second countdown the AI service records the sign for the chosen duration. Use several different people.'),
    table(['Sign', 'Uzbek (Latin)', 'Cyrillic', 'Category', 'Samples', ''], list.filter((s) => s.active).map((s) => [s.name, s.uz_latin, s.uz_cyrillic, s.category,
      el('span', { class: 'pill ' + (s.samples >= 30 ? 'ok' : s.samples >= 10 ? 'warn' : 'bad') }, `${s.samples}${(stats.per_sign[s.id] ? ' · ' + stats.per_sign[s.id].speakers + ' speakers' : '')}`),
      el('div', { class: 'row' }, el('button', { class: 'btn small primary', onclick: () => act(() => api('/api/admin/dataset/record', { method: 'POST', body: { sign_id: s.id, desk_id: Number(deskSel.value), speaker_id: speaker.value || 'unknown', duration: Number(dur.value) } }), '● Recording in 2 s: perform “' + s.name + '”'), }, '● Record'),
        el('button', { class: 'btn small danger', onclick: async () => { if (confirm('Deactivate sign ' + s.name + '?')) { await act(() => api('/api/admin/signs/' + s.id, { method: 'DELETE' })); show(); } } }, '✕'))]))),
  card('Add a new sign class (no code change; retrain afterwards)', el('div', { class: 'inline-form' }, field('Sign name', f.name), field('Uzbek (Latin)', f.latin), field('Cyrillic', f.cyr), field('Category', f.cat), field('Type', f.pos),
    el('button', { class: 'btn primary', onclick: async () => { await act(() => api('/api/admin/signs', { method: 'POST', body: { name: f.name.value, uz_latin: f.latin.value, uz_cyrillic: f.cyr.value || null, category: f.cat.value || 'general', pos: f.pos.value } }), 'Sign added'); show(); } }, 'Add sign')))];
}

async function training() {
  const [models, st] = await Promise.all([api('/api/admin/models'), api('/api/admin/training/status')]);
  const ep = el('input', { type: 'number', value: 60, min: 1, max: 500, style: 'width:100px' });
  const arch = el('select', { style: 'width:auto' }, ['transformer', 'gru', 'tcn'].map((a) => el('option', {}, a)));
  const syn = el('input', { type: 'checkbox', style: 'width:auto' });
  const logBox = el('div', { class: 'log' }, st.log.join('\n') || 'No training run yet.');
  const poll = () => { clearInterval(S.trainTimer); if (st.running) S.trainTimer = setInterval(async () => { if (S.tab !== 'training') return clearInterval(S.trainTimer); const s = await api('/api/admin/training/status'); logBox.textContent = s.log.join('\n'); logBox.scrollTop = logBox.scrollHeight; if (!s.running) { clearInterval(S.trainTimer); show(); } }, 2500); };
  setTimeout(() => { logBox.scrollTop = logBox.scrollHeight; poll(); });
  const metricsBox = el('div');
  return [card('Train / update model', el('div', { class: 'row' }, el('span', {}, 'Epochs'), ep, arch,
    el('label', { class: 'row' }, syn, 'allow SYNTHETIC data (pipeline test only)'),
    el('button', { class: 'btn primary', disabled: st.running, onclick: async () => { await act(() => api('/api/admin/training/start', { method: 'POST', body: { epochs: Number(ep.value), arch: arch.value, allow_synthetic: syn.checked } }), 'Training started'); show(); } }, st.running ? 'Training…' : 'Start training')),
    el('div', { style: 'margin-top:1rem' }, logBox)),
  card('Model versions', table(['Version', 'Arch', 'Classes', 'Val', 'Test', 'Data', 'Active', ''], models.map((m) => [m.version, m.arch, m.num_classes, pct(m.val_accuracy), pct(m.test_accuracy),
    m.synthetic ? el('span', { class: 'pill warn' }, 'SYNTHETIC') : 'real', m.active ? el('span', { class: 'pill ok' }, 'ACTIVE') : '',
    el('div', { class: 'row' }, m.active ? null : el('button', { class: 'btn small', onclick: async () => { await act(() => api(`/api/admin/models/${m.version}/activate`, { method: 'POST' }), 'Model activated; desks reload it automatically'); show(); } }, 'Activate'),
      el('button', { class: 'btn small', onclick: () => showMetrics(m.version, metricsBox) }, 'Metrics'))])), metricsBox)];
}
async function showMetrics(version, box) {
  clear(box);
  const meta = await act(() => api(`/api/admin/models/${version}/metrics`));
  if (!meta) return;
  const res = await fetch(`/api/admin/models/${version}/confusion.png`, { headers: { Authorization: 'Bearer ' + token } });
  const img = res.ok ? el('img', { class: 'confusion', alt: 'confusion matrix', src: URL.createObjectURL(await res.blob()) }) : null;
  const pc = Object.entries(meta.metrics.per_class || {});
  box.append(el('h3', { style: 'margin-top:1.2rem' }, `Metrics for ${version} · split: ${meta.split.mode} · test n=${meta.metrics.test_samples}`),
    el('div', { class: 'two' }, img, table(['Sign', 'Precision', 'Recall', 'Test n'], pc.map(([k, v]) => [k, pct(v.precision), pct(v.recall), v.support]))));
}

async function settings() {
  const s = await api('/api/admin/settings');
  const f = { conf: el('input', { type: 'number', min: 0.3, max: 0.99, step: 0.01, value: s.confidence_threshold }), face: el('input', { type: 'number', min: 0.2, max: 0.99, step: 0.01, value: s.face_match_threshold }),
    idle: el('input', { type: 'number', min: 1, max: 240, value: s.session_idle_minutes }), live: el('input', { type: 'checkbox', checked: s.liveness_required, style: 'width:auto' }) };
  return [card('AI & security settings', el('div', { class: 'inline-form' },
    field('Sign confidence threshold (below = “unclear”)', f.conf), field('Face match threshold', f.face), field('Auto-end idle session (minutes)', f.idle),
    field('Require liveness check', f.live),
    el('button', { class: 'btn primary', onclick: () => act(() => api('/api/admin/settings', { method: 'PUT', body: { confidence_threshold: Number(f.conf.value), face_match_threshold: Number(f.face.value), session_idle_minutes: Number(f.idle.value), liveness_required: f.live.checked } }), 'Saved; desks updated'), }, 'Save')))];
}

async function logs() {
  const [audit, rec, sess] = await Promise.all([api('/api/admin/logs/audit?limit=100'), api('/api/admin/logs/recognition?limit=60'), api('/api/admin/sessions?status=ALL')]);
  return [card('Sessions', table(['#', 'Desk', 'Citizen', 'Auth', 'Status', 'Started', 'Ended'], sess.map((s) => [s.id, s.desk_code, s.citizen ? s.citizen.code : 'guest', s.auth_method, s.status + (s.end_reason ? ' (' + s.end_reason + ')' : ''), fmtDateTime(s.start_time), fmtDateTime(s.end_time)]))),
  card('Audit log', table(['Time', 'Actor', 'Action', 'Target', 'Detail', 'IP'], audit.map((a) => [fmtDateTime(a.ts), a.actor, a.action, a.target, a.detail, a.ip]))),
  card('Recognition log', table(['Time', 'Session', 'Sign', 'Confidence', 'Accepted', 'Model'], rec.map((r) => [fmtDateTime(r.ts), r.session_id || '', r.sign || '—', pct(r.confidence), r.accepted ? '✓' : '✗', r.model])))];
}
const VIEWS = { overview, desks, employees, citizens, signs, training, settings, logs };

// ------------------------------------------------------------------ shell
let content = null;
async function show() {
  if (!content) return;
  clear(content).append(el('div', { class: 'muted' }, '…'));
  try { const kids = await VIEWS[S.tab](); clear(content).append(...kids.filter(Boolean)); }
  catch (e) { clear(content).append(el('div', { class: 'banner bad' }, e.message)); }
}
function render() {
  clear(app);
  if (!token) {
    const u = input('text'), p = el('input', { type: 'password', onkeydown: (e) => { if (e.key === 'Enter') login(u.value, p.value); } });
    app.append(el('div', { class: 'card login' }, el('h2', {}, 'GOV SIGN AI · Admin'), field(t('username'), u), field(t('password'), p),
      S.err ? el('div', { class: 'banner bad', style: 'margin-bottom:1rem' }, S.err) : null, el('button', { class: 'btn primary big', style: 'width:100%', onclick: () => login(u.value, p.value) }, t('login'))));
    return;
  }
  content = el('main');
  app.append(el('div', { class: 'bar' }, el('span', { class: 'brand' }, 'GOV SIGN AI · ADMIN'), el('span', { class: 'spacer' }),
    el('select', { style: 'width:auto', onchange: (e) => { setLang(e.target.value); } }, Object.entries(LANGS).map(([k, l]) => el('option', { value: k, selected: k === getLang() }, l))),
    el('button', { class: 'btn small icon-btn', onclick: toggleTheme }, '◐'), el('span', { class: 'muted' }, user.username), el('button', { class: 'btn small', onclick: logout }, t('logout'))),
  el('div', { class: 'tabs', role: 'tablist' }, TABS.map((k) => el('button', { role: 'tab', 'aria-selected': S.tab === k, onclick: () => { S.tab = k; render(); } }, TITLES[k]))), content);
  show();
}
render();
if (token) openSocket();
