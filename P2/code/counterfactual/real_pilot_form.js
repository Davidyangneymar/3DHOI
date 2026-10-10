(function (root) {
  'use strict';
  const ACTIONS = [['carry','搬运／移动'],['hold','抓持'],['lift','举起'],['place','放置'],['push','推'],['pull','拉'],['other','其他'],['unknown','无法判断']];
  const PHASES = [['approach','接近'],['grasp','建立抓持'],['transport','搬运中'],['sustained_hold','持续抓持'],['placing','放置中'],['release','释放'],['other','其他'],['unknown','无法判断']];
  const TYPES = [['none','未见明确交互错误'],['premature_release','必要抓持提前断开'],['unsupported_motion','缺少必要接触的受控运动'],['incompatible_contact','接触位置不适合当前动作'],['motion_artifact','明显运动／姿态异常'],['other','其他错误'],['uncertain','无法确定错误类型']];
  const REASONS = [['phase_consistent','接触和当前阶段一致'],['required_contact_broken','仍需抓持时接触中断'],['unsupported_controlled_motion','无必要接触或支撑，仍出现受控运动'],['action_incompatible_region','有依据判断该区域不适合当前动作'],['visible_motion_artifact','可见运动或姿态异常'],['insufficient_evidence','缺少足够证据'],['other','其他理由（补充说明）']];
  function makeRecord(entry, state, pilotHash) {
    if (!state.reviewer || !state.reviewer.trim()) throw Error('请填写审核人。');
    if (!['0','1','uncertain'].includes(state.decision)) throw Error('请选择正常、错误或不确定。');
    if (!ACTIONS.some(x=>x[0]===state.action) || !PHASES.some(x=>x[0]===state.phase)) throw Error('请选择动作和阶段。');
    if (state.decision !== 'uncertain' && (state.action==='unknown' || state.phase==='unknown')) throw Error('动作或阶段无法判断时，请选择不确定。');
    if (!/^\d+$/.test(String(state.start)) || !/^\d+$/.test(String(state.end))) throw Error('证据帧必须是整数。');
    const start=Number(state.start), end=Number(state.end);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start<0 || start>end || end>=entry.frames) throw Error('证据范围超出原始帧范围。');
    if (!Array.isArray(state.reasons) || !state.reasons.length || state.reasons.some(v=>!REASONS.some(x=>x[0]===v))) throw Error('请至少选择一个理由。');
    if (state.decision==='0' && state.reasons.some(v=>!['phase_consistent','other'].includes(v))) throw Error('正常结论与所选错误或证据不足理由冲突。');
    if (state.decision==='1' && state.reasons.some(v=>['phase_consistent','insufficient_evidence'].includes(v))) throw Error('确定错误需要一致的语义证据。');
    if (!TYPES.some(x=>x[0]===state.errorType) || (state.decision==='0' && state.errorType!=='none') || (state.decision==='1' && ['none','uncertain'].includes(state.errorType)) || (state.decision==='uncertain' && state.errorType!=='uncertain')) throw Error('错误类型与结论不一致。');
    if (state.decision==='1' && state.legal!=='excluded') throw Error('不能排除合法释放、惯性或外部支撑时，请选不确定。');
    if (([state.action,state.phase,state.errorType,...state.reasons].includes('other') || state.errorType==='incompatible_contact') && !(state.note||'').trim()) throw Error('选择其他或接触区域错误时，请补充具体依据。');
    return {pilot_sha256:pilotHash,sample_id:entry.sample_id,input_sha256:entry.input_sha256,
      reviewer:state.reviewer.trim(),round:'primary',label_error:state.decision==='uncertain'?'uncertain':Number(state.decision),
      action:state.action,phase:state.phase,error_type:state.errorType,reason_codes:state.reasons,
      legal_alternative:state.legal,start_frame:start,end_frame:end,note:(state.note||'').trim(),
      recorded_at:new Date().toISOString(),form_version:'real_pilot_choices_v1',evidence_basis:'user_confirmed_original_frames'};
  }
  function merge(records, incoming, entries, hash) {
    // JSON property order is not part of a review record's meaning.
    function canonical(value) {
      if (Array.isArray(value)) return value.map(canonical);
      if (value && typeof value==='object') return Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])]));
      return value;
    }
    const map=new Map(entries.map(e=>[e.sample_id,e]));
    const result=new Map();
    for (const r of [...records,...incoming]) {
      const e=map.get(r.sample_id);
      if (!e || r.pilot_sha256!==hash || r.input_sha256!==e.input_sha256) throw Error('导入记录来自其他批次或文件。');
      if (r.round!=='primary' || !([0,1,'uncertain'].includes(r.label_error))) throw Error('导入结论或轮次无效。');
      makeRecord(e,{reviewer:r.reviewer,decision:String(r.label_error),action:r.action,phase:r.phase,
        errorType:r.error_type,reasons:r.reason_codes,legal:r.legal_alternative,start:r.start_frame,end:r.end_frame,note:r.note},hash);
      if (result.has(r.sample_id) && JSON.stringify(canonical(result.get(r.sample_id)))!==JSON.stringify(canonical(r))) throw Error('同一样本存在冲突；请保留原记录并人工核对。');
      result.set(r.sample_id,r);
    }
    if (result.size>20 || new Set([...result.values()].map(r=>r.reviewer)).size>1) throw Error('超过预算或混入多个首审人。');
    return [...result.values()];
  }
  const api={ACTIONS,PHASES,TYPES,REASONS,makeRecord,merge};
  if (typeof module==='object' && module.exports) module.exports=api;
  else root.RealPilotForm=api;
})(typeof window==='undefined'?globalThis:window);
