'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const F=require('../counterfactual/real_pilot_form.js');
const source=fs.readFileSync(process.env.REAL_PILOT_APP_SOURCE||path.join(__dirname,'../counterfactual/real_pilot_app.js'),'utf8');
const storageKey='hoi_real_pilot_pilot_state_v2',legacyKey='hoi_real_pilot_pilot';
class Element {
  constructor(){this.value='';this.textContent='';this.children=[];this.files=[];this.checked=false;}
  append(...children){this.children.push(...children);}
  querySelectorAll(selector){const all=this.children.flatMap(c=>[c,...(c.querySelectorAll?.('input')||[])]).filter(c=>c.type==='checkbox');return selector==='input:checked'?all.filter(c=>c.checked):all;}
  click(){}
}
const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
function harness(initial={}) {
  const elements=new Map(),storage=new Map(Object.entries(initial)),renders=[],raf=[],downloads=[];
  const env={failWrite:false,deferRender:false,rejectRender:false,now:0,pending:[]};
  const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const entries=[['A',20,10],['B',4,2]].map(([sample_id,frames,fps])=>({sample_id,frames,fps,input_sha256:sample_id+'hash',condition_text:'test fixture',object:'box',focus:{start:0,end:frames-1},geometry:{joints:Array.from({length:frames},(_,f)=>Array.from({length:24},(_,j)=>[f*.01,j*.01,0]))}}));
  vm.runInNewContext(source,{
    window:{RealPilotForm:F,pilot:{pilot_sha256:'pilot',entries,review_cap:20,failed_count:0,parents:Array(24).fill(-1),meshes:{box:[]}}},
    document:{getElementById:$,createElement:()=>new Element(),createTextNode:t=>({textContent:t})},
    localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>{if(env.failWrite)throw Error('fixture quota exceeded');storage.set(k,v);}},
    Plotly:{react:(id,traces,layout)=>{renders.push({id,traces,layout});if(env.rejectRender)return Promise.reject(Error('fixture render failure'));if(env.deferRender)return new Promise(resolve=>env.pending.push(resolve));return Promise.resolve();}},
    performance:{now:()=>env.now},requestAnimationFrame:cb=>raf.push(cb),
    Blob,URL:{createObjectURL:blob=>{downloads.push(blob);return 'blob:fixture';},revokeObjectURL:()=>{}},setTimeout:()=>{},
  });
  function fill(){for(const [id,value]of Object.entries({reviewer:'test-only',decision:'0',action:'carry',phase:'transport',errorType:'none',legal:'',note:''}))$(id).value=value;for(const c of $('reasons').querySelectorAll('input'))c.checked=c.value==='phase_consistent';}
  function submit(){$('form').onsubmit({preventDefault(){}});}
  async function exported(){$('export').onclick();const text=await downloads.at(-1).text();return text.trim()?text.trim().split('\n').map(JSON.parse):[];}
  return {env,$,storage,renders,raf,entries,fill,submit,exported};
}
test('failed save keeps previous disk and exported record; failed draft prevents navigation',async()=>{
  const h=harness();await flush();h.fill();h.submit();
  const disk=h.storage.get(storageKey),saved=await h.exported();assert.equal(saved.length,1);
  h.env.failWrite=true;h.$('note').value='revision that cannot be saved';h.submit();
  assert.match(h.$('message').textContent,/未保存/);assert.equal(h.storage.get(storageKey),disk);assert.deepEqual(await h.exported(),saved);
  h.$('next').onclick();assert.match(h.$('sample').textContent,/A$/);assert.match(h.$('message').textContent,/草稿未保存/);
});
test('failed import does not change export; successful import commits atomically',async()=>{
  const h=harness();await flush();h.fill();h.submit();const saved=await h.exported();
  const incoming={...saved[0],sample_id:'B',input_sha256:'Bhash',end_frame:3};
  h.$('import').files=[{text:async()=>JSON.stringify(incoming)}];h.env.failWrite=true;await h.$('import').onchange();
  assert.match(h.$('message').textContent,/未导入/);assert.deepEqual(await h.exported(),saved);
  h.env.failWrite=false;await h.$('import').onchange();assert.equal((await h.exported()).length,2);
  assert.equal(JSON.parse(h.storage.get(storageKey)).records.length,2);
});
test('legacy records migrate without deleting keys; saved revisions preserve history',async()=>{
  const empty=harness();await flush();empty.fill();empty.submit();const saved=await empty.exported();
  const original=JSON.stringify(saved),h=harness({[legacyKey]:original});await flush();
  assert.equal(h.$('reviewer').value,'test-only');assert.equal(h.storage.has(storageKey),false);
  h.$('note').value='new confirmed revision';h.submit();const envelope=JSON.parse(h.storage.get(storageKey));
  assert.equal(envelope.records.length,1);assert.equal(envelope.history.length,1);assert.deepEqual(envelope.history[0],saved[0]);assert.equal(h.storage.get(legacyKey),original);
});
test('corrupt storage fails closed and remains untouched',async()=>{
  const h=harness({[storageKey]:'{"version":2,"records":[],"drafts":null,"history":[]}'});await flush();
  const before=[...h.storage];h.fill();h.submit();assert.match(h.$('message').textContent,/存储不可用/);assert.deepEqual([...h.storage],before);
});
test('switching samples while a render awaits cannot resume an old playback loop',async()=>{
  const h=harness();await flush();h.env.deferRender=true;h.$('play').onclick();await flush();assert.equal(h.env.pending.length,1);
  h.$('next').onclick();h.$('play').onclick();
  h.env.deferRender=false;h.env.pending.shift()();await flush();
  assert.equal(h.raf.length,1,'only the new sample may schedule another frame');
  h.env.now=1000;h.raf.shift()();await flush();
  assert.equal(Number(h.$('frame').value),2,'B must use its own 2 FPS');assert.equal(h.renders.at(-1).layout.uirevision,'B');
});
test('pause invalidates queued renders and render failures stop playback',async()=>{
  const h=harness();await flush();h.$('play').onclick();await flush();assert.equal(h.raf.length,1);
  h.$('pause').onclick();h.env.now=1000;h.raf.shift()();await flush();assert.equal(h.raf.length,0);
  h.env.rejectRender=true;h.$('play').onclick();await flush();assert.match(h.$('message').textContent,/回放绘制失败/);assert.equal(h.raf.length,0);
  h.env.rejectRender=false;h.$('play').onclick();await flush();assert.equal(h.raf.length,1,'retry must be possible');
});
