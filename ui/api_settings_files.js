function settingsTransferControls(){return `<section class="panel"><div class="panel-title"><h2>Configuration file</h2></div><div class="button-group"><button class="secondary" data-settings-import>Import env / JSON</button><button class="secondary" data-settings-export>Export saved configuration</button><label class="follow-log"><input id="include-api-keys" type="checkbox">Include API keys (plaintext)</label></div><p class="status-line">Import fills the form. Click Save configuration to apply it.</p></section>`}
function applySettingsFile(data){
 if(data?.format!=='mousegpt-api-settings'||![1,2].includes(data.schemaVersion)||!data.roles||Array.isArray(data.roles))throw Error('Select a MouseGPT API configuration JSON file.');
 const allowed=new Set(['VISION','EMBEDDING','ANALYSIS','AGENT','PHENOTYPE']);
 const fields=new Set(['provider','endpoint','model','api_version','api_key']);
 const configs={...(data.schemaVersion===2?{DEFAULT:data.shared||{}}:{}),...data.roles};
 for(const [role,c] of Object.entries(configs)){
  if((role!=='DEFAULT'&&!allowed.has(role))||!c||typeof c!=='object'||Array.isArray(c))throw Error('Invalid API configuration.');
  for(const [k,v] of Object.entries(c))if(!fields.has(k)||typeof v!=='string')throw Error('Invalid API configuration field.');
  if(c.provider&&!apiProviders.includes(c.provider))throw Error('Unsupported provider: '+c.provider);
 }
 settingsData.shared={...(data.shared||{})};
 settingsData.overrides={};
 for(const role of allowed){settingsData.overrides[role]=!!data.roles[role];if(data.roles[role]){const shared=settingsData.shared,custom=data.roles[role],effective={...shared,...custom};if(role==='EMBEDDING'&&!custom.model)effective.model='text-embedding-3-large';if(custom.provider&&custom.provider!==shared.provider){for(const k of ['endpoint','model','api_version'])if(!(k in custom))delete effective[k];}if(!effective.endpoint)effective.endpoint=envAPIEndpoints[effective.provider||'openai']||'';effective.api_key=custom.api_key||'';effective.api_key_configured=!!custom.api_key;settingsData.roles[role]=effective;}}
 drawSettings();
 toast('Configuration imported. Click Save configuration to apply.');
}
document.addEventListener('click',async e=>{
 const b=e.target.closest('button');if(!b)return;
 try{
  if(b.hasAttribute('data-settings-import'))readAPISettingsFile();
  if(b.hasAttribute('data-settings-export')){
   const data=await api('settings/export',{include_api_keys:!!$('#include-api-keys')?.checked});
   download(data.contains_api_keys?'mousegpt-api-settings-with-keys.json':'mousegpt-api-settings.json',data);
  }
 }catch(err){toast(err.message)}
});

const apiProviders=['openai','azure_openai','openai_compatible','anthropic','gemini','deepseek','openrouter','siliconflow','groq','mistral','xai'];
function apiFields(role,c){return `<div class="row"><label class="field"><span>Provider</span><select data-api="${role}:provider">${apiProviders.map(p=>`<option ${p===(c.provider||'openai')?'selected':''}>${p}</option>`).join('')}</select></label>${[['endpoint','Base URL'],['model',role==='EMBEDDING'?'Embedding model / deployment':'Model / deployment'],['api_version','API version (Azure)']].map(([k,l])=>`<label class="field"><span>${l}</span><input data-api="${role}:${k}" value="${esc(c[k]||'')}"></label>`).join('')}</div><label class="field"><span>API key ${c.api_key_configured?'(configured)':''}</span><input type="password" autocomplete="new-password" data-api="${role}:api_key" value="${esc(c.api_key||'')}" placeholder="Leave blank to retain saved key"></label>`}
const legacyDrawSettings=drawSettings;
drawSettings=function(){
 if(!settingsData||!$('#settings-content'))return;
 legacyDrawSettings();
 const host=$('#settings-content');
 const interpreter=host.querySelector('[data-python]')?.closest('section');
 const existingInterpreter=interpreter?.outerHTML||'';
 host.innerHTML=settingsTransferControls()+panel('Default API',apiFields('DEFAULT',settingsData.shared||{})+'<p class="muted">Used by all roles unless overridden. Embeddings require an embedding model; the default is text-embedding-3-large.</p>')+panel('Role overrides',Object.entries(settingsData.roles).map(([role,c])=>`<section style="margin-bottom:20px"><label class="follow-log"><input type="checkbox" data-api-override="${role}" ${settingsData.overrides?.[role]?'checked':''}>${role} - Custom API</label>${c.error?`<p class="notice">${esc(c.error)} Configure a compatible API for this role before running it.</p>`:''}<div data-role-fields="${role}" ${settingsData.overrides?.[role]?'':'hidden'}>${apiFields(role,c)}</div><button class="secondary" data-api-test="${role}">Save & test ${role}</button></section>`).join(''))+existingInterpreter+'<button class="primary" data-action="save-settings">Save configuration</button>';
 mountTaskMonitors();drawTasks();
};
collectApi=function(){
 const roles={},shared={},interpreters={},inherit=[];
 document.querySelectorAll('[data-api]').forEach(e=>{const [r,k]=e.dataset.api.split(':');if(r==='DEFAULT')shared[k]=e.value;else if(document.querySelector('[data-api-override="'+r+'"]')?.checked)(roles[r]||={})[k]=e.value;});
 document.querySelectorAll('[data-api-override]').forEach(e=>{if(!e.checked)inherit.push(e.dataset.apiOverride)});
 document.querySelectorAll('[data-python]').forEach(e=>{if(e.dataset.dirty==='1')interpreters[e.dataset.python]=e.value});
 return {shared,roles,inherit,interpreters};
};
document.addEventListener('change',e=>{if(e.target.dataset.apiOverride)document.querySelector('[data-role-fields="'+e.target.dataset.apiOverride+'"]').hidden=!e.target.checked;});

const envAPIEndpoints={openai:'https://api.openai.com/v1',anthropic:'https://api.anthropic.com/v1',gemini:'https://generativelanguage.googleapis.com/v1beta/openai',groq:'https://api.groq.com/openai/v1',deepseek:'https://api.deepseek.com',mistral:'https://api.mistral.ai/v1',openrouter:'https://openrouter.ai/api/v1',xai:'https://api.x.ai/v1',siliconflow:'https://api.siliconflow.cn/v1'};
function parseAPIEnv(text){
 const values={},shared={},roles={};let recognized=0;
 for(const [index,raw] of text.replace(/^\uFEFF/,'').split(/\r?\n/).entries()){
  const line=raw.trim();if(!line||line.startsWith('#'))continue;
  const match=line.replace(/^export\s+/,'').match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
  if(!match)throw Error('Invalid env assignment on line '+(index+1)+'.');
  const key=match[1];if(Object.hasOwn(values,key))throw Error('Duplicate env field on line '+(index+1)+'.');
  let value=match[2].trim();
  if(value.startsWith('"')||value.startsWith("'")){
   const quote=value[0],end=value.indexOf(quote,1);
   if(end<1||!/^\s*(#.*)?$/.test(value.slice(end+1)))throw Error('Invalid quoted value on line '+(index+1)+'.');
   value=value.slice(1,end);
  }else value=value.replace(/\s+#.*$/,'').trim();
  values[key]=value;
 }
 const fields=['provider','endpoint','api_key','model','api_version'];
 for(const field of fields){const key='MOUSEGPT_'+field.toUpperCase();if(Object.hasOwn(values,key)){shared[field]=values[key];recognized++;}}
 const fallback=(dest,field,key)=>{if(!dest[field]&&values[key]){dest[field]=values[key];recognized++;}};
 if(!shared.provider)shared.provider=values.AZURE_OPENAI_ENDPOINT?'azure_openai':'openai';
 if(shared.provider==='azure_openai'){
  fallback(shared,'endpoint','AZURE_OPENAI_ENDPOINT');fallback(shared,'api_key','AZURE_OPENAI_API_KEY');fallback(shared,'api_version','AZURE_OPENAI_API_VERSION');
 }else{
  fallback(shared,'api_key','OPENAI_API_KEY');fallback(shared,'endpoint','OPENAI_BASE_URL');fallback(shared,'model','OPENAI_CHAT_MODEL');
 }
 for(const role of ['VISION','EMBEDDING','ANALYSIS','AGENT','PHENOTYPE']){
  const config={};for(const field of fields){const key='MOUSEGPT_'+role+'_'+field.toUpperCase();if(Object.hasOwn(values,key)){config[field]=values[key];recognized++;}}
  if((config.provider||shared.provider)==='azure_openai'){
   const modelKey={VISION:'AZURE_OPENAI_DEPLOYMENT',EMBEDDING:'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',ANALYSIS:'AZURE_OPENAI_CHAT_DEPLOYMENT',AGENT:'AZURE_OPENAI_CHAT_DEPLOYMENT'}[role];
   if(modelKey)fallback(config,'model',modelKey);
   const versionKey={EMBEDDING:'AZURE_OPENAI_EMBEDDING_API_VERSION',ANALYSIS:'AZURE_OPENAI_CHAT_API_VERSION',AGENT:'AZURE_OPENAI_CHAT_API_VERSION'}[role];
   if(versionKey)fallback(config,'api_version',versionKey);
  }
  if(Object.keys(config).length)roles[role]=config;
 }
 if(shared.provider==='azure_openai'){
  if(!shared.model)shared.model=roles.ANALYSIS?.model||roles.AGENT?.model||roles.VISION?.model||'';
  if(!shared.api_version)shared.api_version=roles.ANALYSIS?.api_version||roles.AGENT?.api_version||roles.VISION?.api_version||'';
 }
 if(!recognized)throw Error('No supported API configuration fields found.');
 if(!shared.endpoint)shared.endpoint=envAPIEndpoints[shared.provider]||'';
 return {format:'mousegpt-api-settings',schemaVersion:2,shared,roles};
}
function readAPISettingsFile(){
 const input=document.createElement('input');input.type='file';input.accept='.json,.env,.local,.example,.txt';
 input.onchange=async()=>{try{const file=input.files[0];if(!file)return;if(file.size>1024*1024)throw Error('Configuration file exceeds 1 MB.');const text=await file.text();applySettingsFile(text.trimStart().startsWith('{')?JSON.parse(text):parseAPIEnv(text));}catch(e){toast(e.message)}};
 input.click();
}
