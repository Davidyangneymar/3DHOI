'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const F=require('../counterfactual/real_pilot_form.js');
const entry={sample_id:'fixture',input_sha256:'hash',frames:20};
const state={reviewer:'test-only',decision:'0',action:'carry',phase:'transport',start:'0',end:'19',errorType:'none',reasons:['phase_consistent'],legal:'',note:''};
test('blank decisions and semantic fields never create a label',()=>{
 for(const key of ['decision','action','phase','errorType','reviewer'])assert.throws(()=>F.makeRecord(entry,{...state,[key]:''},'pilot'));
});
test('normal, error and uncertain need only choices when evidence is sufficient',()=>{
 assert.equal(F.makeRecord(entry,state,'pilot').label_error,0);
 assert.equal(F.makeRecord(entry,{...state,decision:'1',errorType:'unsupported_motion',legal:'excluded',reasons:['unsupported_controlled_motion']},'pilot').label_error,1);
 assert.equal(F.makeRecord(entry,{...state,decision:'uncertain',action:'unknown',phase:'unknown',errorType:'uncertain',reasons:['insufficient_evidence']},'pilot').label_error,'uncertain');
});
test('legitimate alternatives and out-of-range evidence prevent definite error',()=>{
 for(const legal of ['','possible','unknown'])assert.throws(()=>F.makeRecord(entry,{...state,decision:'1',errorType:'premature_release',legal},'pilot'));
 for(const start of [true,'',-1,20,1.2])assert.throws(()=>F.makeRecord(entry,{...state,start},'pilot'));
});
test('imports validate sample identity, conclusion and conflicts',()=>{
 const r=F.makeRecord(entry,state,'pilot');assert.equal(F.merge([r],[r],[entry],'pilot').length,1);
 for(const change of [{pilot_sha256:'other'},{input_sha256:'other'},{label_error:true},{end_frame:100},{round:'secondary'}])assert.throws(()=>F.merge([],[{...r,...change}],[entry],'pilot'));
 assert.throws(()=>F.merge([r],[{...r,note:'conflict'}],[entry],'pilot'));
});
test('duplicate imports ignore property order but retain revision conflicts',()=>{
 const r=F.makeRecord(entry,state,'pilot');
 const reordered=Object.fromEntries(Object.entries(r).reverse());
 assert.equal(F.merge([r],[reordered],[entry],'pilot').length,1);
 assert.throws(()=>F.merge([r],[{...reordered,recorded_at:'different revision'}],[entry],'pilot'));
});
test('uncertain counts toward cap and cannot introduce sample 21',()=>{
 const entries=Array.from({length:21},(_,i)=>({...entry,sample_id:String(i)}));
 const records=entries.map(e=>F.makeRecord(e,{...state,decision:'uncertain',errorType:'uncertain',reasons:['insufficient_evidence']},'pilot'));
 assert.equal(F.merge([],records.slice(0,20),entries,'pilot').length,20);
 assert.throws(()=>F.merge([],records,entries,'pilot'));
});
test('reasons must agree with conclusion and region errors require specific evidence',()=>{
 assert.throws(()=>F.makeRecord(entry,{...state,reasons:['unsupported_controlled_motion']},'pilot'));
 const s={...state,decision:'1',errorType:'incompatible_contact',reasons:['action_incompatible_region'],legal:'excluded'};
 assert.throws(()=>F.makeRecord(entry,s,'pilot'));
 assert.equal(F.makeRecord(entry,{...s,note:'test-only concrete region explanation'},'pilot').label_error,1);
});
