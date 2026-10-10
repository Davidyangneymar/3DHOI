/* Pure form contract, shared by the offline page and Node unit tests. */
(function (root) {
  'use strict';
  const ACTIONS = [
    ['lift', '抬起 / 提起'], ['carry', '搬运 / 持物移动'], ['hold', '原地抓持 / 支撑'],
    ['place', '放下 / 放置'], ['push', '推动'], ['pull', '拉动'],
    ['rotate', '旋转 / 调整朝向'], ['clean', '拖地 / 吸尘等工具操作'],
    ['handover', '换手 / 重新抓持'], ['release', '释放 / 投掷'],
    ['unknown', '无法判断'], ['other', '其他（补充说明）']
  ];
  const PHASES = [
    ['approach', '接近 / 准备接触'], ['grasp', '建立抓持'], ['sustained_hold', '持续抓持'],
    ['transport', '搬运 / 移动中'], ['supported', '接触外部支撑 / 放置中'],
    ['regrasp', '换手 / 调整抓持'], ['release', '释放接触'], ['after_release', '释放之后'],
    ['unknown', '无法判断'], ['other', '其他（补充说明）']
  ];
  const REASONS = {
    error: [
      ['required_contact_broken', '需要持续抓持时，接触提前断开'],
      ['unsupported_motion', '缺少必要接触，物体仍保持受控运动'],
      ['action_incompatible_region', '接触位置不适合当前动作（需补充具体位置）'],
      ['other', '其他错误（补充说明）']
    ],
    valid: [
      ['contact_matches_phase', '接触与当前动作阶段一致'],
      ['legitimate_regrasp', '属于合理换手 / 调整抓持'],
      ['legitimate_release', '属于合理释放 / 释放后运动'],
      ['visible_support', '存在可见的合理支撑'],
      ['other', '其他正常依据（补充说明）']
    ],
    uncertain: [
      ['phase_unclear', '无法确定动作或交互阶段'],
      ['release_possible', '可能是合理释放 / 惯性运动'],
      ['support_possible', '无法排除外部支撑'],
      ['region_unclear', '无法判断这个接触位置是否合理'],
      ['view_insufficient', '当前视角或几何信息不足'],
      ['context_missing', '缺少文字或上下文信息'],
      ['other', '其他不确定因素（补充说明）']
    ],
    invalid: [
      ['pose_artifact', '人体姿态 / 骨骼有明显异常'],
      ['geometry_artifact', '物体几何 / 变换异常'],
      ['alignment_artifact', '人体与物体坐标未对齐'],
      ['animation_artifact', '明显跳变 / 拼接等构造问题'],
      ['other', '其他无效原因（补充说明）']
    ]
  };
  const LEGALITY = [
    ['excluded', '已结合该片段，排除合理释放、惯性和外部支撑等解释'],
    ['possible', '仍有可能是合法交互'],
    ['unknown', '信息不足，无法排除']
  ];
  const text = value => typeof value === 'string' ? value.trim() : '';
  function selection(options, code, custom) {
    const found = options.find(([value]) => value === code);
    if (!found) return '';
    return code === 'other' ? text(custom) : code;
  }
  function reasonLabels(decision, codes) {
    const options = REASONS[decision] || [];
    return codes.map(code => options.find(([id]) => id === code)?.[1]).filter(Boolean);
  }
  function buildRecord(entry, state, reviewedAt) {
    const reviewer = text(state.reviewer), decision = state.decision;
    if (!reviewer) throw Error('请先填写审核人；只需填写一次。');
    if (!['primary', 'secondary'].includes(state.round)) throw Error('请选择审核轮次。');
    if (!REASONS[decision]) throw Error('请选择一个审核结论。');
    if (decision === 'valid' && entry.type !== 'clean') throw Error('扰动可能合理时请选择「不确定」，不自动改为正常例。');
    const action = selection(ACTIONS, state.actionCode, state.actionOther);
    const phase = selection(PHASES, state.phaseCode, state.phaseOther);
    const definitive = ['error', 'valid'].includes(decision);
    if (definitive && (!action || !phase || state.actionCode === 'unknown' || state.phaseCode === 'unknown')) {
      throw Error('确定判定需要明确动作和阶段；无法判断时请选择「不确定」。');
    }
    if (state.actionCode === 'other' && !action || state.phaseCode === 'other' && !phase) throw Error('选择「其他」时，请补充对应的动作或阶段。');
    const start = state.start === '' || state.start == null ? null : Number(state.start);
    const end = state.end === '' || state.end == null ? null : Number(state.end);
    if ((start !== null && (!Number.isInteger(start) || start < 0 || start >= entry.frames)) ||
        (end !== null && (!Number.isInteger(end) || end < 0 || end >= entry.frames)) ||
        (start !== null && end !== null && start > end)) throw Error('证据帧段无效：起点必须早于或等于终点，且在本序列范围内。');
    if (definitive && (start === null || end === null)) throw Error('请用当前帧标记证据起点和终点，或选择「整段」。');
    const codes = [...new Set(state.reasonCodes || [])];
    const allowed = new Set(REASONS[decision].map(([code]) => code));
    if (!codes.length || codes.some(code => !allowed.has(code))) throw Error('请至少选择一个与当前结论相符的理由。');
    const note = text(state.note);
    const spatial = decision === 'error' && (entry.type === 'spatial_contact_shift' || codes.includes('action_incompatible_region'));
    if ((codes.includes('other') || spatial) && !note) throw Error(spatial ? '请简短说明哪个接触位置不适合当前动作；仅仅位置不同不足以判错。' : '选择「其他」时，请补充一句理由。');
    if (decision === 'error' && state.legality !== 'excluded') throw Error('仍不能排除合法解释时，请选择「不确定」。');
    const reasons = reasonLabels(decision, codes);
    if (decision === 'error') reasons.push(LEGALITY[0][1]);
    if (note) reasons.push('补充：' + note);
    return {sample_id: entry.sample_id, sample_sha256: entry.sha256, reviewer, round: state.round,
      decision, action, phase, start_frame: start, end_frame: end, reason: reasons.join('；'),
      reviewed_at: reviewedAt, form_version: 'choices_v2', reason_codes: codes,
      legal_alternatives: decision === 'error' ? state.legality : null, note,
      action_choice: state.actionCode || '', phase_choice: state.phaseCode || ''};
  }
  function restoreRecord(record) {
    if (!record) return {actionCode:'',actionOther:'',phaseCode:'',phaseOther:'',start:'',end:'',decision:'',reasonCodes:[],legality:'',note:''};
    const actionCode = record.action_choice || (ACTIONS.some(([id]) => id === record.action) ? record.action : record.action ? 'other' : '');
    const phaseCode = record.phase_choice || (PHASES.some(([id]) => id === record.phase) ? record.phase : record.phase ? 'other' : '');
    return {actionCode, actionOther:actionCode === 'other' ? record.action : '',
      phaseCode, phaseOther:phaseCode === 'other' ? record.phase : '',
      start:record.start_frame ?? '', end:record.end_frame ?? '', decision:record.decision,
      reasonCodes:record.form_version === 'choices_v2' ? [...(record.reason_codes || [])] : ['other'],
      legality:record.legal_alternatives || '', note:record.form_version === 'choices_v2' ? record.note || '' : record.reason || ''};
  }
  function prepareEvidence(state, range, enabled, hasPrior) {
    const result = {...state};
    if (enabled && !hasPrior && (state.start === '' || state.start == null) && (state.end === '' || state.end == null) &&
        range && Number.isInteger(range.start) && Number.isInteger(range.end) && range.start >= 0 && range.end >= range.start) {
      result.start = range.start; result.end = range.end; result.evidenceOrigin = 'auto_focus_pending_confirmation';
    }
    return result;
  }
  function key(record) { return JSON.stringify([record.sample_id, record.reviewer, record.round]); }
  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, stable(value[k])]));
    return value;
  }
  function mergeRecords(existing, incoming, entries) {
    const bySample = new Map(entries.map(e => [e.sample_id, e]));
    const result = existing.slice();
    for (const record of incoming) {
      const entry = bySample.get(record.sample_id);
      if (!entry || entry.sha256 !== record.sample_sha256) throw Error('导入样本不属于本批，或样本哈希不一致。');
      if (!text(record.reviewer) || !text(record.reason) || !['primary', 'secondary'].includes(record.round) || !REASONS[record.decision]) throw Error('导入记录字段不完整。');
      if (record.decision === 'valid' && entry.type !== 'clean') throw Error('扰动样本不能直接导入为正常例。');
      if (['error','valid'].includes(record.decision) && (!text(record.action) || !text(record.phase) || !Number.isInteger(record.start_frame) || !Number.isInteger(record.end_frame) || record.start_frame < 0 || record.end_frame < record.start_frame || record.end_frame >= entry.frames)) throw Error('导入的确定判定缺少有效动作、阶段或证据帧段。');
      const previous = result.find(r => key(r) === key(record));
      if (previous && JSON.stringify(stable(previous)) !== JSON.stringify(stable(record))) throw Error('同一判定存在不同版本。请先处理冲突，原记录未被覆盖。');
      if (!previous) result.push(record);
    }
    return result;
  }
  function exportRecords(records, entries, reviewer, round) {
    const hashes = new Map(entries.map(e => [e.sample_id, e.sha256]));
    return records.filter(r => hashes.get(r.sample_id) === r.sample_sha256 && r.reviewer === reviewer && r.round === round);
  }
  const api = {ACTIONS,PHASES,REASONS,LEGALITY,buildRecord,restoreRecord,prepareEvidence,key,mergeRecords,exportRecords};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewForm = api;
})(globalThis);
