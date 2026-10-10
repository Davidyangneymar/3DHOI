(function () {
  'use strict';
  const $ = id => document.getElementById(id), F = window.ReviewForm, entries = queue.entries;
  const context = queue.review_context || {};
  const storageKey = context.storage_key || 'hoi_semantic_review_20261001';
  window.samples = {};
  let records = [], drafts = {}, writable = true, position = 0, token = 0, active, clean;
  let timer = null, timelineReady = false, rendering = false, redraw = false;
  let identity = {reviewer:'', round:'primary'};
  let evidenceOrigin = 'manual';
  function announce(message, error = false) {
    $('message').textContent = message;
    $('message').className = error ? 'error' : 'success';
  }
  try {
    records = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (!Array.isArray(records) || records.some(r => !r || typeof r.sample_id !== 'string')) throw Error('记录格式异常');
  } catch (error) {
    writable = false;
    announce('无法读取旧记录，已停止写入以保留原数据：' + error.message, true);
  }
  try {
    drafts = JSON.parse(localStorage.getItem(storageKey + '_drafts_v2') || '{}');
    if (!drafts || typeof drafts !== 'object' || Array.isArray(drafts)) throw Error('草稿格式异常');
    const remembered = JSON.parse(localStorage.getItem(storageKey + '_identity_v2') || 'null');
    if (remembered && typeof remembered.reviewer === 'string' && ['primary','secondary'].includes(remembered.round)) identity = remembered;
  } catch (error) { drafts = {}; announce('未能恢复草稿或审核人设置：' + error.message, true); }
  function options(id, rows) {
    const blank = document.createElement('option'); blank.value = ''; blank.textContent = '请选择';
    $(id).append(blank);
    for (const [value, label] of rows) {
      const option = document.createElement('option'); option.value = value; option.textContent = label; $(id).append(option);
    }
  }
  if (context.required_reviewer) identity.reviewer = context.required_reviewer;
  if (context.required_round) identity.round = context.required_round;
  options('action', F.ACTIONS); options('phase', F.PHASES); options('legality', F.LEGALITY);
  $('reviewer').value = identity.reviewer; $('round').value = identity.round;
  $('reviewer').disabled = Boolean(context.required_reviewer);
  $('round').disabled = Boolean(context.required_round);
  if (context.title) {
    document.title = context.title;
    document.querySelector('h1').textContent = context.title;
    const guide = document.createElement('p'); guide.className = 'notice'; guide.textContent = context.instructions;
    document.querySelector('.workspace').before(guide);
  }
  const previousPanel = document.createElement('p'); previousPanel.className = 'notice'; previousPanel.hidden = true;
  $('review-form').before(previousPanel);
  const focusPanel = document.createElement('div'); focusPanel.className = 'notice'; focusPanel.hidden = true;
  const focusText = document.createElement('span'), focusButton = document.createElement('button');
  focusButton.type = 'button'; focusButton.textContent = '跳到定位片段';
  const focusOnly = document.createElement('input'); focusOnly.type = 'checkbox';
  const focusLabel = document.createElement('label'); focusLabel.append(focusOnly, ' 只播放定位片段');
  const useFocus = document.createElement('button'); useFocus.type = 'button'; useFocus.textContent = '用此片段作证据';
  focusPanel.append(focusText, document.createElement('br'), focusButton, focusLabel, useFocus);
  $('scene').after(focusPanel);
  focusButton.onclick = () => { if (!active) return; stop(); $('frame').value = entry().review_focus.start; focusOnly.checked = true; draw(); };
  useFocus.onclick = () => { if (!active) return; $('start').value = entry().review_focus.start; $('end').value = entry().review_focus.end; evidenceOrigin = 'user_selected_focus'; hints(); stash(); };
  const evidenceSummary = document.createElement('p'); evidenceSummary.className = 'notice';
  const mechanicsText = document.createElement('pre');
  if (context.semantic_only) {
    const field = $('start').closest('fieldset'), details = document.createElement('details'), summary = document.createElement('summary');
    summary.textContent = '自动检查、客观线索与证据帧（可展开修改）';
    field.before(evidenceSummary,details); details.append(summary,mechanicsText,field);
    field.querySelector('legend').textContent = '调整证据片段';
    document.querySelector('input[name="decision"]').closest('fieldset').querySelector('.number').textContent = '2';
    $('reason-options').closest('fieldset').querySelector('.number').textContent = '3';
    $('save').textContent = '确认片段并保存'; $('save-next').textContent = '确认并下一条 →';
  }
  function checkContext(record) {
    if (context.required_reviewer && record.reviewer !== context.required_reviewer) throw Error('此入口仅用于原审核人复核自己的首轮记录。');
    if (context.required_round && record.round !== context.required_round) throw Error('此入口的审核轮次不匹配。');
    if ((context.excluded_reviewers || []).includes(record.reviewer)) throw Error('独立二审需由另一位审核人完成，请使用其稳定代号。');
  }
  const entry = () => entries[position];
  const formKey = () => F.key({sample_id:entry().sample_id, ...identity});
  const saved = e => records.find(r => r.sample_id === e.sample_id && r.sample_sha256 === e.sha256 && r.reviewer === identity.reviewer && r.round === identity.round);
  const decision = () => document.querySelector('input[name="decision"]:checked')?.value || '';
  const reasonCodes = () => [...$('reason-options').querySelectorAll('input:checked')].map(input => input.value);
  function state() {
    return {...identity, actionCode:$('action').value, actionOther:$('action-other').value,
      phaseCode:$('phase').value, phaseOther:$('phase-other').value, start:$('start').value, end:$('end').value,
      decision:decision(), reasonCodes:reasonCodes(), legality:$('legality').value, note:$('note').value, evidenceOrigin};
  }
  function reasons(selected = []) {
    $('reason-options').replaceChildren();
    const rows = F.REASONS[decision()] || [];
    if (!rows.length) $('reason-options').textContent = '选择结论后显示对应理由。';
    for (const [value, text] of rows) {
      const label = document.createElement('label'), input = document.createElement('input'), span = document.createElement('span');
      label.className = 'reason'; input.type = 'checkbox'; input.value = value; input.checked = selected.includes(value);
      span.textContent = text; label.append(input, span); $('reason-options').append(label);
    }
  }
  function hints() {
    $('action-other').hidden = $('action').value !== 'other';
    $('phase-other').hidden = $('phase').value !== 'other';
    $('legality-wrap').hidden = decision() !== 'error';
    const spatial = decision() === 'error' && (entry().type === 'spatial_contact_shift' || reasonCodes().includes('action_incompatible_region'));
    $('spatial-hint').hidden = !spatial;
    $('note-label').textContent = spatial || reasonCodes().includes('other') ? '补充说明（必填一句）' : '补充说明（选填）';
    if (context.semantic_only) evidenceSummary.textContent = '证据片段：' + ($('start').value || '未设定') + '–' + ($('end').value || '未设定') + ' 帧。系统已准备候选范围，保存即确认该范围支持你的判定；不合适可展开修改。动作、阶段、结论与理由仍由你选择。';
  }
  function restore() {
    const draft = drafts[formKey()], record = saved(entry());
    const hasDraft = draft?.sha256 === entry().sha256;
    const priorState = hasDraft ? draft.state : {...F.restoreRecord(record), evidenceOrigin:record?.evidence_selection?.origin || 'manual'};
    const s = F.prepareEvidence(priorState,entry().review_focus,context.semantic_only,Boolean(hasDraft || record));
    evidenceOrigin = s.evidenceOrigin || 'manual';
    for (const [id, key] of [['action','actionCode'],['action-other','actionOther'],['phase','phaseCode'],['phase-other','phaseOther'],['start','start'],['end','end'],['legality','legality'],['note','note']]) $(id).value = s[key] ?? '';
    document.querySelectorAll('input[name="decision"]').forEach(input => { input.checked = input.value === s.decision; });
    $('valid-choice').disabled = entry().type !== 'clean';
    reasons(s.reasonCodes || []); hints();
    $('draft-status').textContent = draft?.sha256 === entry().sha256 ? '已恢复草稿，尚未保存为正式判定。' : record ? '已读取此审核人本轮保存的判定。修改后请重新保存。' : '尚未审核；不会预选结论或理由。';
    updateProgress();
  }
  function stash() {
    const nextDrafts = {...drafts, [formKey()]:{sha256:entry().sha256, state:state()}};
    try {
      localStorage.setItem(storageKey + '_drafts_v2', JSON.stringify(nextDrafts)); drafts = nextDrafts;
      $('draft-status').textContent = '草稿已自动保留，尚未保存为正式判定。';
    } catch (error) { announce('草稿未能保留，请先导出已保存记录：' + error.message, true); }
  }
  function updateProgress() {
    const count = entries.filter(saved).length;
    $('progress').textContent = identity.reviewer ? identity.reviewer + ' · ' + (identity.round === 'primary' ? '第一轮' : '第二轮') + '：已保存 ' + count + ' / ' + entries.length + ' 条' : '请先填写审核人；共 ' + entries.length + ' 条。';
    $('progress-bar').max = entries.length; $('progress-bar').value = count;
    [...$('sample').options].forEach((option, i) => { option.textContent = (saved(entries[i]) ? '✓ ' : '') + (i + 1) + '. ' + entries[i].sample_id; });
    $('prev').disabled = position === 0; $('next').disabled = position === entries.length - 1;
  }
  function changeIdentity() {
    // The form still belongs to the previous identity until this assignment.
    stash(); identity = {reviewer:$('reviewer').value.trim(), round:$('round').value};
    try { localStorage.setItem(storageKey + '_identity_v2', JSON.stringify(identity)); }
    catch (error) { announce('审核人设置未能保留：' + error.message, true); }
    restore();
  }
  $('reviewer').onchange = changeIdentity; $('round').onchange = changeIdentity;
  $('review-form').addEventListener('input', event => {
    if (event.target.name === 'decision') reasons();
    if (['start','end'].includes(event.target.id)) evidenceOrigin = 'user_selected_range';
    hints(); stash();
  });
  // Some browsers emit only change for select controls.
  $('review-form').addEventListener('change', event => {
    if (event.target.tagName === 'SELECT') { hints(); stash(); }
  });
  entries.forEach((e, i) => { const option = document.createElement('option'); option.value = i; $('sample').append(option); });
  const loads = new Map();
  function load(id) {
    if (window.samples[id]) return Promise.resolve(window.samples[id]);
    if (loads.has(id)) return loads.get(id);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script'); script.src = 'samples/' + encodeURIComponent(id) + '.js';
      script.onload = () => window.samples[id] ? resolve(window.samples[id]) : reject(Error('样本文件缺少数据：' + id));
      script.onerror = () => reject(Error('无法加载样本：' + id)); document.head.append(script);
    });
    loads.set(id, promise); promise.catch(() => loads.delete(id)); return promise;
  }
  function points(a, name, color, mode = 'markers', size = 2) {
    return {x:a.map(p => p[0]), y:a.map(p => p[1]), z:a.map(p => p[2]), type:'scatter3d', mode, name,
      marker:{size, color}, line:{width:5, color}, hoverinfo:'name'};
  }
  function geometry(sample, e, frame, color, name) {
    const traces = [];
    for (const part of queue.meshes[e.object]) {
      const p = part.prefix, r = sample[p + '_rot_mat'][frame], t = sample[p + '_trans'][frame], scale = sample[p + '_scale'][frame];
      const xyz = part.vertices.map(v => [0,1,2].map(i => scale * (r[i][0]*v[0] + r[i][1]*v[1] + r[i][2]*v[2]) + t[i]));
      traces.push(points(xyz, name + '物体', color));
    }
    const joints = sample.joints[frame], lines = [];
    queue.parents.forEach((p, i) => { if (p >= 0) lines.push(joints[p], joints[i], [null,null,null]); });
    traces.push(points(lines, name + '人体', color, 'lines'));
    if (name === '当前') traces.push(points([joints[22],joints[23]], '当前手部', '#e99c3f', 'markers', 5));
    return traces;
  }
  async function draw() {
    redraw = true;
    if (rendering) return;
    rendering = true;
    try {
      while (redraw) {
        redraw = false;
        if (!active) continue;
        const e = entry(), frame = +$('frame').value, my = token;
        $('frame-label').textContent = frame + ' / ' + (e.frames - 1);
        const traces = [...($('show-clean').checked ? geometry(clean,e,frame,'#b9c2c8','原始') : []), ...geometry(active,e,frame,'#11899a','当前')];
        await Plotly.react('scene', traces, {margin:{l:0,r:0,t:5,b:0}, showlegend:false, uirevision:e.source_sequence,
          scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},zaxis:{title:'Z (m)'}}}, {responsive:true,displaylogo:false});
        if (my === token) $('scene-status').textContent = '';
      }
    } catch (error) { stop(); $('scene-status').textContent = '回放绘制失败：' + error.message; }
    finally { rendering = false; }
  }
  function stop() { clearTimeout(timer); timer = null; $('play').textContent = '▶ 播放'; }
  function tick() {
    if (!active) return stop();
    const next = +$('frame').value + 1;
    const last = focusOnly.checked && entry().review_focus ? entry().review_focus.end : entry().frames - 1;
    if (next > last) return stop();
    $('frame').value = next; draw(); timer = setTimeout(tick, 120);
  }
  $('play').onclick = () => {
    if (timer !== null) return stop();
    if (!active) return;
    const range = focusOnly.checked && entry().review_focus ? entry().review_focus : {start:0,end:entry().frames-1};
    if (+$('frame').value >= range.end || +$('frame').value < range.start) { $('frame').value = range.start; draw(); }
    $('play').textContent = 'Ⅱ 暂停'; timer = setTimeout(tick, 120);
  };
  $('frame').oninput = () => {
    stop();
    const range = entry().review_focus;
    if (range && (+$('frame').value < range.start || +$('frame').value > range.end)) focusOnly.checked = false;
    draw();
  };
  $('show-clean').onchange = draw;
  async function timeline() {
    if (!active || !$('technical').open || timelineReady) return;
    timelineReady = true;
    const traces = [];
    [clean,active].forEach((sample,k) => [0,1].forEach(h => traces.push({y:sample.distances_m.map(p => p[h]),name:(k?'当前':'原始')+(h?'右手':'左手'),line:{dash:k?'solid':'dash'}})));
    try {
      await Plotly.react('timeline', traces, {margin:{t:15,b:40,l:50,r:10},xaxis:{title:'帧'},yaxis:{title:'最近顶点距离 (m)'},legend:{orientation:'h'},shapes:[{type:'line',x0:0,x1:entry().frames-1,y0:.05,y1:.05,line:{dash:'dot'}}]}, {responsive:true,displaylogo:false});
    } catch (error) { timelineReady = false; announce('距离曲线加载失败：' + error.message, true); }
  }
  $('technical').ontoggle = timeline;
  async function choose(index) {
    stop(); position = index; $('sample').value = index;
    const my = ++token, e = entry(); active = null; clean = null; timelineReady = false;
    focusPanel.hidden = !e.review_focus;
    focusOnly.checked = Boolean(e.review_focus);
    if (e.review_focus) focusText.textContent = e.review_focus.description + ' 定位帧段 ' + e.review_focus.start + '–' + e.review_focus.end + '；这是导航线索，不是错误证据。拖动进度条可查看完整序列。';
    for (const id of ['play','frame','mark-start','mark-end']) $(id).disabled = true;
    $('scene').style.visibility = 'hidden'; $('timeline').style.visibility = 'hidden';
    $('scene-status').textContent = '正在加载回放…'; $('frame-label').textContent = '—';
    $('sample-title').textContent = e.sample_id; $('sample-badge').textContent = e.second_review_required ? '需独立二审' : '分歧时需二审';
    $('description').textContent = '原始文本：' + (e.description || '缺失，请结合回放判断；不要据此推定动作。');
    $('parameters').textContent = '旧构造类型：' + e.type + '\n旧构造参数：' + JSON.stringify(e.parameters,null,2);
    if (context.semantic_only) {
      const m = e.review_mechanics;
      mechanicsText.textContent = m ? '自动核对：原始样本哈希一致；帧范围有效；坐标和距离有限值。\n统计范围：'+m.range.start+'–'+m.range.end+' 帧（不是改判依据）。\n左右手到最近网格顶点距离中位数：'+m.hand_median_distance_m.map(v=>v.toFixed(3)).join(' / ')+' 米。\n双手距离均 ≥ 0.05 米：'+m.both_hands_far_frames+' / '+m.frame_count+' 帧。\n物体平移路径长：'+m.object_translation_path_m.toFixed(3)+' 米；单步最大平移：'+m.object_max_translation_step_m.toFixed(3)+' 米/帧。\n这些统计不能证明接触、穿透、外部支撑或运动是否合理；没有自动生成语义标签。' : '本页没有预计算客观统计，请结合回放判断。';
    }
    $('start').max = e.frames - 1; $('end').max = e.frames - 1; restore();
    const previous = context.previous_records?.[e.sample_id];
    previousPanel.hidden = !previous;
    if (previous) previousPanel.textContent = '你上次选择“样本无效”。原说明：' + previous.note + '；原证据帧：' + previous.start_frame + '–' + previous.end_frame + '。请重新判断分类，可保留原结论，但需说明具体构造异常；也可以选择不确定。此处不预选新结论。';
    try {
      const loaded = await Promise.all([load(e.sample_id),load(e.source_sequence + '__clean__none')]);
      if (my !== token) return;
      [active,clean] = loaded;
      $('frame').max = e.frames - 1; $('frame').value = previous?.start_frame ?? e.review_focus?.start ?? 0;
      for (const id of ['play','frame','mark-start','mark-end']) $(id).disabled = false;
      $('scene').style.visibility = ''; $('timeline').style.visibility = ''; draw(); timeline();
    } catch (error) { if (my === token) $('scene-status').textContent = '加载失败：' + error.message + '。请检查本地样本文件。'; }
  }
  $('sample').onchange = () => choose(+$('sample').value);
  $('prev').onclick = () => choose(Math.max(0,position-1));
  $('next').onclick = () => choose(Math.min(entries.length-1,position+1));
  function nextPending() {
    for (let step = 1; step <= entries.length; step++) {
      const index = (position + step) % entries.length;
      if (!saved(entries[index])) { choose(index); return true; }
    }
    announce('本批当前审核人、当前轮次已全部保存，请导出记录交回。'); return false;
  }
  $('next-pending').onclick = nextPending;
  for (const [button, input] of [['mark-start','start'],['mark-end','end']]) $(button).onclick = () => { if (!active) return; $(input).value = $('frame').value; evidenceOrigin = 'user_selected_range'; hints(); stash(); };
  $('whole').onclick = () => { $('start').value = 0; $('end').value = entry().frames - 1; evidenceOrigin = 'user_selected_whole_sequence'; hints(); stash(); };
  function persist(next) {
    if (!writable) throw Error('旧记录未能安全读取，不能写入。');
    localStorage.setItem(storageKey, JSON.stringify(next)); records = next;
  }
  function save(advance) {
    try {
      if (!writable) throw Error('旧记录未能安全读取，不能写入。');
      if (context.semantic_only && !active) throw Error('请等待回放成功加载后再确认判定。');
      const record = F.buildRecord(entry(),state(),new Date().toISOString());
      if (context.semantic_only) {
        record.evidence_selection = {origin:evidenceOrigin === 'auto_focus_pending_confirmation' ? 'auto_focus_confirmed_on_save' : evidenceOrigin,
          confirmed_at:record.reviewed_at,start_frame:record.start_frame,end_frame:record.end_frame};
        if (entry().review_mechanics) record.automatic_checks = entry().review_mechanics;
      }
      checkContext(record);
      if (context.packet_id) record.review_packet_id = context.packet_id;
      const prior = context.previous_records?.[entry().sample_id];
      if (prior) record.supersedes_record_sha256 = prior.record_sha256;
      const previous = records.find(r => F.key(r) === F.key(record));
      if (previous) {
        const history = JSON.parse(localStorage.getItem(storageKey + '_history_v2') || '[]');
        if (!Array.isArray(history)) throw Error('历史记录格式异常，未覆盖原判定。');
        history.push(previous); localStorage.setItem(storageKey + '_history_v2',JSON.stringify(history));
      }
      persist([...records.filter(r => F.key(r) !== F.key(record)),record]);
      delete drafts[formKey()];
      let warning = '';
      try { localStorage.setItem(storageKey + '_drafts_v2',JSON.stringify(drafts)); }
      catch (error) { warning = '（草稿清理失败，刷新后可能仍显示旧草稿。）'; }
      restore(); announce('已保存当前判定。完成后请导出 JSONL。' + warning);
      if (advance) nextPending();
    } catch (error) { announce(error.message,true); }
  }
  $('save').onclick = () => save(false);
  $('review-form').onsubmit = event => { event.preventDefault(); save(true); };
  $('export').onclick = () => {
    const rows = F.exportRecords(records,entries,identity.reviewer,identity.round);
    if (!rows.length) return announce('本批当前审核人、当前轮次尚无已保存记录。',true);
    const blob = new Blob([rows.map(r => JSON.stringify(r)).join('\n') + '\n'], {type:'application/jsonl'});
    const link = document.createElement('a'), url = URL.createObjectURL(blob);
    link.href = url; link.download = 'semantic_reviews_' + identity.reviewer.replace(/[^\w\u4e00-\u9fff-]/g,'_') + '_' + identity.round + (context.packet_id ? '_' + context.packet_id : '') + '.jsonl';
    link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
    announce('已发起导出 ' + rows.length + ' 条正式记录，请检查下载文件。');
  };
  $('import').onchange = async () => {
    try {
      const file = $('import').files[0]; if (!file) return;
      const allIncoming = (await file.text()).split(/\r?\n/).filter(s => s.trim()).map(s => JSON.parse(s));
      const incoming = context.import_subset ? allIncoming.filter(r => entries.some(e => e.sample_id === r.sample_id)) : allIncoming;
      if (!incoming.length) throw Error('文件中没有属于当前小批次的记录。');
      for (const record of incoming) {
        checkContext(record);
        if (context.packet_id && ![context.packet_id,...(context.accepted_packet_ids || [])].includes(record.review_packet_id)) throw Error('请导入本入口或明确兼容批次导出的记录。');
      }
      const merged = F.mergeRecords(records,incoming,entries); persist(merged); restore();
      announce('已导入 ' + incoming.length + ' 条记录。' + (allIncoming.length > incoming.length ? '另有 ' + (allIncoming.length - incoming.length) + ' 条属于其他小批次，未导入此页。' : '') + '原文件未修改。');
    } catch (error) { announce('导入失败：' + error.message,true); }
    finally { $('import').value = ''; }
  };
  if (entries.length) choose(0);
  else { writable = false; announce('审核队列为空。',true); }
})();
