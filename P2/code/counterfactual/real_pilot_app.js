(function(){
  'use strict';
  const $=id=>document.getElementById(id), F=window.RealPilotForm, P=window.pilot, entries=P.entries;
  const key='hoi_real_pilot_'+P.pilot_sha256;
  const storageKey=key+'_state_v2';
  let position=0, frame=0, playing=false, range=null, records=[], drafts={}, history=[], writable=true, drawSerial=Promise.resolve(), playEpoch=0, renderEpoch=0;
  function message(s,error=false){$('message').textContent=s;$('message').className=error?'error':'';}
  try {
    const saved=localStorage.getItem(storageKey);
    const data=saved===null?{version:2,records:JSON.parse(localStorage.getItem(key)||'[]'),drafts:JSON.parse(localStorage.getItem(key+'_drafts')||'{}'),history:JSON.parse(localStorage.getItem(key+'_history')||'[]')}:JSON.parse(saved);
    if (!data || data.version!==2 || !Array.isArray(data.records) || !Array.isArray(data.history) || !data.drafts || typeof data.drafts!=='object' || Array.isArray(data.drafts)) throw Error('存储格式无效。');
    const validRecords=F.merge([],data.records,entries,P.pilot_sha256);
    for (const r of data.history) F.merge([],[r],entries,P.pilot_sha256);
    for (const [id,s] of Object.entries(data.drafts)) {
      if (!entries.some(e=>e.sample_id===id) || !s || typeof s!=='object' || Array.isArray(s) || (s.reasons!==undefined && (!Array.isArray(s.reasons) || s.reasons.some(r=>typeof r!=='string')))) throw Error('草稿格式无效。');
    }
    records=validRecords;drafts=data.drafts;history=data.history;
  }
  catch(e){writable=false;message('无法读取旧记录，已停止写入以保留原数据：'+e.message,true);}
  function persist(nextRecords=records,nextDrafts=drafts,nextHistory=history){
    if(!writable)throw Error('存储不可用，未保存；请保留原数据。');
    // A single setItem is atomic. Commit memory only after durable storage succeeds.
    localStorage.setItem(storageKey,JSON.stringify({version:2,records:nextRecords,drafts:nextDrafts,history:nextHistory}));
    records=nextRecords;drafts=nextDrafts;history=nextHistory;
  }
  function options(id,rows){const blank=document.createElement('option');blank.value='';blank.textContent='请选择';$(id).append(blank);for(const [v,t]of rows){const el=document.createElement('option');el.value=v;el.textContent=t;$(id).append(el);}}
  options('action',F.ACTIONS);options('phase',F.PHASES);options('errorType',F.TYPES);
  for(const [v,t]of F.REASONS){const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.value=v;l.append(c,document.createTextNode(t));$('reasons').append(l);}
  const inputs=['reviewer','action','phase','decision','errorType','legal','start','end','note'];
  function state(){return Object.fromEntries(inputs.map(id=>[id,$(id).value]).concat([['reasons',[...$('reasons').querySelectorAll('input:checked')].map(e=>e.value)]]));}
  function entry(){return entries[position];}
  function stash(){if(!entry())return true;if(!writable)return false;try{persist(records,{...drafts,[entry().sample_id]:state()});return true;}catch(e){message('草稿未保存，请先保留当前填写内容：'+e.message,true);return false;}}
  function progress(){ $('progress').textContent=`已保存 ${records.length}/${entries.length} · 上限 ${P.review_cap} · 生成失败 ${P.failed_count} 条另记`;$('previous').disabled=position===0;$('next').disabled=position>=entries.length-1;}
  function restore(){const e=entry(),r=records.find(r=>r.sample_id===e.sample_id);const s=drafts[e.sample_id]||(r?{reviewer:r.reviewer,action:r.action,phase:r.phase,decision:String(r.label_error),errorType:r.error_type,legal:r.legal_alternative,start:r.start_frame,end:r.end_frame,note:r.note,reasons:r.reason_codes}:{reviewer:records[0]?.reviewer||'',start:e.focus.start,end:e.focus.end});for(const id of inputs)$(id).value=s[id]??'';for(const c of $('reasons').querySelectorAll('input'))c.checked=(s.reasons||[]).includes(c.value);}
  function objectPoints(e,f){const points=[];for(const part of P.meshes[e.object]){const g=e.geometry,r=g[part.prefix+'_rot_mat'][f],s=g[part.prefix+'_scale'][f],t=g[part.prefix+'_trans'][f];for(const v of part.vertices)points.push(t.map((x,a)=>x+s*(r[a][0]*v[0]+r[a][1]*v[1]+r[a][2]*v[2])));}return points;}
  function trace(points,mode,color,size){return {type:'scatter3d',mode,x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),marker:{color,size},line:{color,width:5},hoverinfo:'skip',showlegend:false};}
  function ranges(e){const points=e.geometry.joints.flat();for(const f of [0,Math.floor(e.frames/2),e.frames-1])points.push(...objectPoints(e,f));const lows=[Infinity,Infinity,Infinity],highs=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let a=0;a<3;a++){lows[a]=Math.min(lows[a],p[a]);highs[a]=Math.max(highs[a],p[a]);}const half=Math.max(.5,...highs.map((h,a)=>(h-lows[a])*.55));return lows.map((lo,a)=>[(lo+highs[a])/2-half,(lo+highs[a])/2+half]);}
  let axes=null;
  function draw(){const e=entry(),f=frame,epoch=renderEpoch;if(!e)return Promise.resolve();$('frame').value=f;$('frameText').textContent=`帧 ${f}/${e.frames-1} · ${(f/e.fps).toFixed(2)} 秒`;
    const joints=e.geometry.joints[f],bones=[];P.parents.forEach((p,i)=>{if(p>=0)bones.push(joints[p],joints[i],null);});
    const traces=[trace(objectPoints(e,f),'markers','#7196ab',2),trace(bones,'lines','#273d53',2),trace([joints[22]],'markers','#ed7849',7),trace([joints[23]],'markers','#2577d4',7)];
    const layout={margin:{l:0,r:0,t:0,b:0},uirevision:e.sample_id,scene:{aspectmode:'cube',xaxis:{title:'X (m)',range:axes[0]},yaxis:{title:'Y (m)',range:axes[1]},zaxis:{title:'Z (m)',range:axes[2]}}};
    drawSerial=drawSerial.catch(()=>{}).then(()=>{if(epoch===renderEpoch)return Plotly.react('scene',traces,layout,{responsive:true,displaylogo:false});});return drawSerial;
  }
  function drawOnce(){const epoch=renderEpoch;draw().catch(e=>{if(epoch===renderEpoch)message('回放绘制失败，请重试：'+e.message,true);});}
  function stop(){playing=false;playEpoch++;renderEpoch++;}
  function load(){stop();const e=entry();if(!e){$('form').hidden=true;$('sample').textContent='暂无可审核输出';message('当前批次无成功生成结果，失败已记录；不会自动补采样。');progress();return;}
    $('sample').textContent=`${position+1}. ${e.sample_id}`;$('condition').textContent=`生成条件：${e.condition_text}；物体：${e.object}。请按实际回放判断动作。`;
    $('focus').textContent=`定位帧 ${e.focus.start}–${e.focus.end}，依据手部运动变化，不代表检测到错误。原始 FPS：${e.fps}。`;
    range=[e.focus.start,e.focus.end];frame=range[0];$('frame').max=e.frames-1;axes=ranges(e);restore();progress();drawOnce();}
  $('previous').onclick=()=>{if(stash()&&position>0){position--;load();}};$('next').onclick=()=>{if(stash()&&position<entries.length-1){position++;load();}};
  $('frame').oninput=()=>{stop();frame=Number($('frame').value);drawOnce();};$('pause').onclick=stop;
  $('jump').onclick=()=>{if(!entry())return;stop();range=[entry().focus.start,entry().focus.end];frame=range[0];drawOnce();};
  $('full').onclick=()=>{if(!entry())return;stop();range=[0,entry().frames-1];frame=0;drawOnce();};
  $('play').onclick=()=>{
    if(!entry()||playing)return;
    playing=true;const epoch=++playEpoch;
    if(frame>=range[1])frame=range[0];
    const start=frame,t=performance.now(),e=entry(),end=range[1];
    async function tick(){
      if(!playing||epoch!==playEpoch)return;
      frame=Math.min(end,start+Math.floor((performance.now()-t)*e.fps/1000));
      try{await draw();}catch(error){if(epoch===playEpoch){stop();message('回放绘制失败，请重试：'+error.message,true);}return;}
      if(!playing||epoch!==playEpoch)return;
      if(frame>=end)stop();else requestAnimationFrame(tick);
    }
    tick();
  };
  $('useFrame').onclick=()=>{$('start').value=frame;stash();};$('form').oninput=stash;
  $('form').onsubmit=ev=>{ev.preventDefault();try{if(!writable)throw Error('存储不可用，未保存。');const record=F.makeRecord(entry(),state(),P.pilot_sha256),previous=records.find(r=>r.sample_id===record.sample_id);const updated=F.merge(records.filter(r=>r.sample_id!==record.sample_id),[record],entries,P.pilot_sha256);persist(updated,{...drafts,[record.sample_id]:state()},previous?[...history,previous]:history);progress();message('已保存。可切换下一条，结束后导出记录。');}catch(e){message('未保存：'+e.message,true);}};
  $('export').onclick=()=>{const blob=new Blob([records.map(r=>JSON.stringify(r)).join('\n')+(records.length?'\n':'')],{type:'application/x-ndjson'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='real_error_pilot_primary_'+P.pilot_sha256.slice(0,10)+'.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import').onchange=async()=>{try{const file=$('import').files[0];if(!file)return;const incoming=(await file.text()).replace(/^\uFEFF/,'').split(/\r?\n/).filter(s=>s.trim()).map(s=>JSON.parse(s));const updated=F.merge(records,incoming,entries,P.pilot_sha256);persist(updated);load();message('导入成功。');}catch(e){message('未导入：'+e.message,true);}};
  load();
})();
