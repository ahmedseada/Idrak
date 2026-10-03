// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands.Serve;

/// <summary>The web chat page served at <c>/ui</c>: one HTML file that streams answers from the OpenAI-style API.</summary>
internal static class ChatPage
{
    public const string Html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Idrak chat</title>
<style>
  :root { --bg: #fafafa; --fg: #1b1b1b; --muted: #6b6b6b; --line: #dddddd; --me: #e8eefc; --it: #ffffff; --accent: #2f5bd3; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #ececec; --muted: #9a9a9a; --line: #333333; --me: #23314f; --it: #1f1f1f; --accent: #7aa0ff; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
  header { display: flex; gap: 8px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
  header strong { margin-right: auto; }
  select, input, textarea, button { font: inherit; color: inherit; background: var(--it); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
  button { background: var(--accent); color: #fff; border: none; cursor: pointer; }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 760px; white-space: pre-wrap; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--it); }
  .user { align-self: flex-end; background: var(--me); }
  .stats { color: var(--muted); font-size: 12px; }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--line); }
  textarea { flex: 1; resize: vertical; min-height: 44px; }
</style>
</head>
<body>
<header>
  <strong>Idrak chat</strong>
  <select id="model" aria-label="Model"></select>
  <input id="key" type="password" placeholder="API key (if set)" aria-label="API key">
  <button id="reset" type="button">New chat</button>
</header>
<div id="log"></div>
<form id="form">
  <textarea id="prompt" placeholder="Message (Enter sends, Shift+Enter for a new line)"></textarea>
  <button>Send</button>
</form>
<script>
const log = document.getElementById('log'), prompt = document.getElementById('prompt'), model = document.getElementById('model'), key = document.getElementById('key');
let messages = [];
const headers = () => Object.assign({ 'Content-Type': 'application/json' }, key.value ? { Authorization: 'Bearer ' + key.value } : {});
function add(cls, text) { const d = document.createElement('div'); d.className = 'msg ' + cls; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
async function models() {
  const r = await fetch('v1/models', { headers: headers() });
  if (!r.ok) { add('stats', 'Could not list the models (' + r.status + '); enter the API key if the server has one.'); return; }
  model.innerHTML = '';
  for (const m of (await r.json()).data) { const o = document.createElement('option'); o.textContent = m.id; model.appendChild(o); }
}
async function send(text) {
  messages.push({ role: 'user', content: text });
  add('user', text);
  const out = add('it', ''), started = performance.now();
  let answer = '', usage = null;
  const r = await fetch('v1/chat/completions', { method: 'POST', headers: headers(),
    body: JSON.stringify({ model: model.value, messages, stream: true, stream_options: { include_usage: true } }) });
  if (!r.ok) { out.textContent = 'Error ' + r.status + ': ' + await r.text(); return; }
  const reader = r.body.getReader(), decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const line = buffer.slice(0, cut).replace(/^data: /, ''); buffer = buffer.slice(cut + 2);
      if (line === '[DONE]') continue;
      const chunk = JSON.parse(line);
      if (chunk.usage) usage = chunk.usage;
      const delta = chunk.choices[0] && chunk.choices[0].delta;
      if (delta && delta.content) { answer += delta.content; out.textContent = answer; log.scrollTop = log.scrollHeight; }
    }
  }
  messages.push({ role: 'assistant', content: answer });
  const seconds = (performance.now() - started) / 1000;
  if (usage) add('stats', usage.completion_tokens + ' tokens in ' + seconds.toFixed(1) + ' s (' + (usage.completion_tokens / seconds).toFixed(1) + ' tokens/s)');
}
document.getElementById('form').addEventListener('submit', e => { e.preventDefault(); const t = prompt.value.trim(); if (t) { prompt.value = ''; send(t); } });
prompt.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('form').requestSubmit(); } });
document.getElementById('reset').addEventListener('click', () => { messages = []; log.innerHTML = ''; });
key.addEventListener('change', models);
models();
</script>
</body>
</html>
""";
}
