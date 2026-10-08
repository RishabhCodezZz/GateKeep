const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
let corpora = [];
let busy = false;
let stop = null;

const here = () => corpora.find((c) => c.ready);

async function load() {
  corpora = (await (await fetch("/api/info")).json()).corpora;
  const missing = corpora.filter((c) => c.prepare);
  $("#hint").hidden = !missing.length;
  $("#hint").textContent = missing.map((c) => `Not prepared: ${c.prepare}`).join(" · ");
  showExamples();
  syncSend();
  if (!here()) $("#examples").replaceChildren(el("p", "note", "Nothing is ready yet. Prepare a source with the command shown under the top bar."));
}

function showExamples() {
  const box = $("#examples");
  box.replaceChildren();
  $("#sub").textContent = here() ? `Ask about a concept from ${here().label}.` : "";
  for (const q of here()?.examples ?? []) {
    const b = el("button", "", q);
    b.type = "button";
    b.onclick = () => send(q);
    box.append(b);
  }
}

const MARKUP = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\\\(.+?\\\)|\$(?!\s)[^$\n]+?(?<!\s)\$|\[\d+\])/;

// TeX in answers: powers and indices become real superscripts and subscripts, the rest stays as written
function tex(node, src) {
  for (const part of src.split(/(\^\{[^}]*\}|_\{[^}]*\}|\^[\w+-]|_[\w+-])/)) {
    if (/^[\^_]/.test(part)) node.append(el(part[0] === "^" ? "sup" : "sub", "", part.slice(1).replace(/^\{|\}$/g, "")));
    else if (part) node.append(part);
  }
  return node;
}

function inline(parent, text, open) {
  for (const part of text.split(MARKUP)) {
    const cite = part.match(/^\[(\d+)\]$/);
    if (cite) { const b = el("button", "cite", part); b.type = "button"; b.setAttribute("aria-label", "Source " + cite[1]); b.dataset.src = cite[1] - 1; b.setAttribute("aria-expanded", "false"); b.onclick = () => open(Number(cite[1]) - 1); parent.append(b); }
    else if (/^\*\*.+\*\*$/.test(part)) parent.append(el("strong", "", part.slice(2, -2)));
    else if (/^\*[^*]+\*$/.test(part)) parent.append(el("em", "", part.slice(1, -1)));
    else if (/^`.+`$/.test(part)) parent.append(el("code", "", part.slice(1, -1)));
    else if (/^\\\(.+\\\)$/.test(part)) parent.append(tex(el("code", "math"), part.slice(2, -2).trim()));
    else if (/^\$.+\$$/.test(part)) parent.append(tex(el("code", "math"), part.slice(1, -1)));
    else if (part) parent.append(part);
  }
}

// ponytail: bold, italic, code, bullet lists and LaTeX shown as monospace; a real maths renderer (KaTeX) if formulas matter
function renderAnswer(box, text, open) {
  box.replaceChildren();
  for (const seg of text.split(/(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/)) {
    if (seg.startsWith("\\[") || seg.startsWith("$$")) { box.append(tex(el("pre", "math"), seg.slice(2, -2).trim())); continue; }
    for (const para of seg.split(/\n{2,}/)) {
      let ul = null, p = null;
      for (const line of para.trim().split("\n")) {
        if (!line.trim()) continue;
        const item = line.match(/^\s*[-*]\s+(.*)/);
        if (item) { p = null; if (!ul) { ul = el("ul"); box.append(ul); } const li = el("li"); ul.append(li); inline(li, item[1], open); }
        else { ul = null; if (p) p.append("\n"); else { p = el("p"); box.append(p); } inline(p, line, open); }
      }
    }
  }
}

const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const morph = (change) => (document.startViewTransition && !reduced.matches ? document.startViewTransition(change) : change());
let follow = true;  // stop auto-scrolling once the reader scrolls up
addEventListener("scroll", () => { follow = innerHeight + scrollY >= document.documentElement.scrollHeight - 120; });

function newTurn(question, label) {
  const li = el("li", "turn");
  const answer = el("div", "a pending", "Checking… ");
  const clock = el("span", "clock", "0 s");
  answer.append(clock);
  answer.setAttribute("role", "status");
  answer.setAttribute("aria-busy", "true");
  const t0 = Date.now();
  const tick = setInterval(() => { clock.textContent = Math.round((Date.now() - t0) / 1000) + " s"; }, 1000);
  const settle = () => { clearInterval(tick); answer.setAttribute("aria-busy", "false"); };
  const chips = el("div", "chips");
  const trace = el("details", "trace");
  trace.open = true;
  const strip = el("span", "strip");
  const verdict = el("span", "", "How this answer was checked");
  const summary = el("summary", "verdict");
  summary.append(strip, verdict);
  trace.append(summary);
  const rows = el("ol");
  trace.append(rows);
  li.append(el("p", "q", question), answer, chips, trace);
  $("#chat").append(li);
  let items = [], shown = -1, checks = 0, byGemma = 0;
  const open = (i) => {
    li.querySelector(".passage")?.remove();
    shown = shown === i ? -1 : i;
    li.querySelectorAll("[data-src]").forEach((b) => b.setAttribute("aria-expanded", String(Number(b.dataset.src) === shown)));
    const p = items[shown];
    if (!p) return;
    const box = el("div", "passage");
    box.append(el("span", "meta", [p.source, p.section, p.page != null ? "page " + p.page : ""].filter(Boolean).join(" · ") || "passage " + (i + 1)), p.text);
    chips.after(box);
  };
  const scroll = () => { if (follow) li.scrollIntoView({ block: "end", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); };
  return {
    event(ev) {
      if (ev.type === "step") rows.append(el("li", "step", ev.text));
      else if (ev.type === "gate") {
        checks++;
        const gemma = ev.who.includes("Gemma");
        byGemma += gemma;
        strip.append(el("i", gemma ? "gemma" : "laya"));
        const row = el("li", "gate", `${ev.gate}: ${ev.n === 1 ? ev.labels[0] : ev.labels.filter((l) => l === "yes").length + " of " + ev.n + " yes"}`);
        row.append(el("span", "badge " + (gemma ? "gemma" : "laya"), ev.who), el("span", "ms", ev.ms + " ms"));
        rows.append(row);
      } else if (ev.type === "passages") {
        items = ev.items;
        items.forEach((p, i) => { const b = el("button", "", `[${i + 1}] ${p.source || ""}`.trim()); b.type = "button"; b.setAttribute("aria-label", `Source ${i + 1}${p.source ? ", " + p.source : ""}`); b.dataset.src = i; b.setAttribute("aria-expanded", "false"); b.onclick = () => open(i); chips.append(b); });
      } else if (ev.type === "answer") {
        settle();
        answer.className = "a" + (ev.refused ? " refused" : "");
        if (!ev.refused) renderAnswer(answer, ev.text, open);
        else answer.textContent = ev.reason === "off_topic" ? "Not about machine learning, so I didn't look it up."
          : `No passage in ${label} answered this (query rewritten twice).`;
      } else if (ev.type === "done") {
        const s = ev.seconds.toFixed(1) + " s";
        verdict.textContent = !checks ? s : byGemma ? `Laya made ${checks - byGemma} of ${checks} checks, Gemma ${byGemma} · ${s}` : `Laya made all ${checks} check${checks === 1 ? "" : "s"} · ${s}`;
        trace.open = false;
      } else if (ev.type === "error") this.error(ev.message);
      scroll();
    },
    error(msg) {
      settle(); answer.hidden = true;
      const box = el("div", "error");
      box.setAttribute("role", "alert");
      const again = el("button", "secondary", "Retry");
      again.type = "button";
      again.onclick = () => send(question);
      box.append("Something went wrong: " + msg + " ", again);
      li.append(box); scroll();
    },
    stopped() { settle(); answer.className = "a refused"; answer.textContent = "Stopped."; },
  };
}

async function send(question) {
  question = question.trim();
  if (!question || busy || !here()) return;
  busy = true; follow = true; stop = new AbortController();
  morph(() => document.body.removeAttribute("data-empty"));
  $("#question").value = ""; grow(); syncSend();
  const turn = newTurn(question, here().label);
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, signal: stop.signal,
      body: JSON.stringify({ question, mode: "auto", corpus: here().name }) });
    if (!res.ok) { turn.error((await res.json()).error); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (line) turn.event(JSON.parse(line)); }
    }
    if (buf.trim()) turn.event(JSON.parse(buf));
  } catch (e) {
    if (e.name === "AbortError") turn.stopped(); else turn.error(String(e));
  } finally {
    busy = false; syncSend(); $("#question").focus();
  }
}

$("#ask").addEventListener("submit", (e) => { e.preventDefault(); if (busy) stop.abort(); else send($("#question").value); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($("#question").value); } });
$("#new-chat").addEventListener("click", () => { if (!busy) morph(() => { $("#chat").replaceChildren(); document.body.setAttribute("data-empty", ""); }); });
const root = document.documentElement;
document.querySelectorAll("input[name=theme]").forEach((r) => {
  r.checked = r.value === root.dataset.theme;
  r.onchange = () => { root.dataset.theme = r.value; try { localStorage.setItem("theme", r.value); } catch (e) {} };
});
// no transitions until the first frame is on screen, so nothing slides into place when the page opens
requestAnimationFrame(() => requestAnimationFrame(() => root.setAttribute("data-ready", "")));
function syncSend() {
  const b = $("#send");
  b.classList.toggle("stop", busy);
  b.setAttribute("aria-label", busy ? "Stop" : "Send");
  b.disabled = !busy && (!$("#question").value.trim() || !here());
}
function grow() { const t = $("#question"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; }
$("#question").addEventListener("input", () => { grow(); syncSend(); });
load();
