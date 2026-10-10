// Synthetic fixtures only: these tests do not produce human review records.
'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const F = require('../counterfactual/review_form.js');
const clean = {sample_id:'synthetic_clean',sha256:'test-hash',type:'clean',frames:20};
const perturbed = {...clean,sample_id:'synthetic_error',type:'premature_release'};
const spatial = {...perturbed,type:'spatial_contact_shift'};
const normal = {reviewer:'test-reviewer',round:'primary',actionCode:'carry',phaseCode:'transport',
  actionOther:'',phaseOther:'',start:'2',end:'15',decision:'valid',reasonCodes:['contact_matches_phase'],note:'',legality:''};
const bad = {...normal,decision:'error',reasonCodes:['required_contact_broken'],legality:'excluded'};
const make = (e,s) => F.buildRecord(e,s,'2000-01-01T00:00:00Z');
test('automatic evidence fills only the range, never semantic choices', () => {
  const original=F.restoreRecord(), copy=JSON.stringify(original);
  const s=F.prepareEvidence(original,{start:12,end:30},true,false);
  assert.equal(s.start,12); assert.equal(s.end,30);
  assert.equal(s.decision,''); assert.equal(s.actionCode,''); assert.equal(s.phaseCode,'');
  assert.deepEqual(s.reasonCodes,[]); assert.equal(JSON.stringify(original),copy);
});
test('existing records, drafts and manual evidence are never replaced by defaults', () => {
  const blank=F.restoreRecord(), range={start:12,end:30};
  assert.deepEqual(F.prepareEvidence(blank,range,true,true),blank);
  assert.deepEqual(F.prepareEvidence(blank,range,false,false),blank);
  const manual={...blank,start:4,end:8};
  assert.deepEqual(F.prepareEvidence(manual,range,true,false),manual);
  assert.deepEqual(F.prepareEvidence(blank,{start:5,end:2},true,false),blank);
});
test('blank form never invents a label', () => {
  assert.equal(F.restoreRecord().decision,'');
  assert.deepEqual(F.restoreRecord().reasonCodes,[]);
  assert.throws(() => make(clean,{...normal,decision:''}));
  assert.throws(() => make(clean,{...normal,reviewer:' '}));
});
test('normal clean example requires no free text', () => {
  const r = make(clean,normal);
  assert.equal(r.decision,'valid'); assert.equal(r.action,'carry'); assert.equal(r.phase,'transport');
  assert.equal(r.start_frame,2); assert.equal(r.end_frame,15); assert.ok(r.reason.includes('接触与当前动作阶段一致'));
  assert.equal(r.form_version,'choices_v2'); assert.equal(r.sample_sha256,clean.sha256);
});
test('perturbation cannot be saved as a normal example', () => {
  assert.throws(() => make(perturbed,normal),/不确定/);
});
test('error requires explicit rejection of legitimate alternatives', () => {
  for (const legality of ['', 'possible','unknown']) assert.throws(() => make(perturbed,{...bad,legality}),/合法解释/);
  assert.equal(make(perturbed,bad).decision,'error');
});
test('spatial error requires specific written evidence regardless of selected reason', () => {
  assert.throws(() => make(spatial,bad),/接触位置/);
  assert.throws(() => make(perturbed,{...bad,reasonCodes:['action_incompatible_region']}),/接触位置/);
  assert.ok(make(spatial,{...bad,note:'Synthetic evidence for testing only.'}).reason.includes('Synthetic evidence'));
});
test('uncertainty supports unknown action and empty evidence range', () => {
  const r = make(spatial,{...normal,decision:'uncertain',actionCode:'unknown',phaseCode:'unknown',start:'',end:'',reasonCodes:['phase_unclear']});
  assert.equal(r.decision,'uncertain'); assert.equal(r.start_frame,null); assert.equal(r.end_frame,null);
});
test('definitive decisions cannot use unknown actions or missing ranges', () => {
  assert.throws(() => make(clean,{...normal,actionCode:'unknown'}));
  assert.throws(() => make(clean,{...normal,phaseCode:''}));
  assert.throws(() => make(clean,{...normal,start:''}));
});
test('invalid examples can be reported without inferred actions', () => {
  const r = make(clean,{...F.restoreRecord(),reviewer:'test-reviewer',round:'primary',decision:'invalid',reasonCodes:['pose_artifact']});
  assert.equal(r.action,''); assert.equal(r.decision,'invalid');
});
test('other action, phase and reason require explanation', () => {
  for (const change of [{actionCode:'other'},{phaseCode:'other'},{reasonCodes:['other']}]) assert.throws(() => make(clean,{...normal,...change}));
  const r = make(clean,{...normal,actionCode:'other',actionOther:'custom action',phaseCode:'other',phaseOther:'custom phase',reasonCodes:['other'],note:'custom reason'});
  assert.equal(r.action,'custom action'); assert.equal(r.phase,'custom phase');
});
test('frame ranges must be finite integers within sample and ordered', () => {
  for (const [start,end] of [['-1','5'],['2.5','5'],['3','20'],['8','4'],['NaN','5'],['0','Infinity']]) assert.throws(() => make(clean,{...normal,start,end}));
});
test('reasons must match the verdict and cannot be silently invented', () => {
  for (const reasonCodes of [[],['not_a_reason'],['required_contact_broken']]) assert.throws(() => make(clean,{...normal,reasonCodes}));
  assert.equal(make(clean,{...normal,reasonCodes:['contact_matches_phase','contact_matches_phase']}).reason_codes.length,1);
});
test('legacy free text restores without losing original wording', () => {
  const record = {action:'搬椅子',phase:'持续抓持中段',reason:'原始人工理由',start_frame:0,end_frame:5,decision:'error'};
  const copy = JSON.stringify(record), s = F.restoreRecord(record);
  assert.equal(s.actionCode,'other'); assert.equal(s.actionOther,record.action);
  assert.equal(s.phaseCode,'other'); assert.equal(s.phaseOther,record.phase);
  assert.equal(s.note,record.reason); assert.deepEqual(s.reasonCodes,['other']); assert.equal(JSON.stringify(record),copy);
});
test('choice records survive save and restore', () => {
  const r = make(clean,normal), restored = F.restoreRecord(r);
  assert.deepEqual(make(clean,{...restored,reviewer:normal.reviewer,round:normal.round}),r);
});
test('same record with different property ordering merges once', () => {
  const r = make(clean,normal), reordered = Object.fromEntries(Object.entries(r).reverse());
  assert.equal(F.mergeRecords([r],[reordered],[clean]).length,1);
});
test('conflicting import fails atomically without modifying existing records', () => {
  const r = make(clean,normal), existing = [r], before = JSON.stringify(existing);
  assert.throws(() => F.mergeRecords(existing,[make(perturbed,bad),{...r,reason:'conflict'}],[clean,perturbed]),/不同版本/);
  assert.equal(JSON.stringify(existing),before);
});
test('imports reject unknown samples, changed hashes and incomplete definitive labels', () => {
  const r = make(clean,normal);
  for (const change of [{sample_id:'unknown'},{sample_sha256:'changed'},{start_frame:null},{end_frame:20},{action:''},{reason:''}]) assert.throws(() => F.mergeRecords([],[{...r,...change}],[clean]));
});
test('export is limited to current reviewer, round and batch hash', () => {
  const r = make(clean,normal);
  const rows = [r,{...r,reviewer:'another'},{...r,round:'secondary'},make(perturbed,bad),{...r,sample_sha256:'old'}];
  assert.deepEqual(F.exportRecords(rows,[clean],normal.reviewer,'primary'),[r]);
});
