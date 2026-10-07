const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
let corpora = [];
let busy = false;

const picked = (name) => document.querySelector(`input[name=${name}]:checked`)?.value;

async function load() {
  corpora = (await (await fetch("/api/info")).json()).corpora;
  const box = $("#corpus");
  const first = corpora.find((c) => c.ready);
  for (const c of corpora) {
    const label = el("label", c.ready ? "" : "off");
    const input = Object.assign(document.createElement("input"), { type: "radio", name: "corpus", value: c.name, disabled: !c.ready, checked: c === first });
    label.append(input, " " + c.label);
    box.append(label);
    if (!c.ready) box.append(el("div", "hint", "Not prepared: " + c.prepare));
  }
  box.addEventListener("change", showExamples);
  showExamples();
  if (!first) {
    $("#send").disabled = true;
    const examplesBox = $("#examples");
    examplesBox.replaceChildren(el("p", "note", "No corpus is ready yet. Prepare one with the command shown in the sidebar."));
  }
}

function showExamples() {
  const box = $("#examples");
  box.replaceChildren();
  for (const q of corpora.find((c) => c.name === picked("corpus"))?.examples ?? []) {
    const b = el("button", "", q);
    b.type = "button";
    b.onclick = () => send(q);
    box.append(b);
  }
}

function renderAnswer(box, text, open) {
  box.replaceChildren();
  for (const para of text.split(/\n{2,}/)) {
    const p = el("p");
    for (const part of para.split(/(\[\d+\])/)) {
      const m = part.match(/^\[(\d+)\]$/);
      if (m) { const b = el("button", "cite", part); b.type = "button"; b.setAttribute("aria-label", "Source " + m[1]); b.onclick = () => open(Number(m[1]) - 1); p.append(b); }
      else p.append(part);
    }
    box.append(p);
  }
}

function newTurn(question, label) {
  const li = el("li", "turn");
  const answer = el("div", "a pending", "Checking…");
  answer.setAttribute("role", "status");
  const chips = el("div", "chips");
  const trace = el("details", "trace");
  trace.open = true;
  trace.append(el("summary", "", "How this answer was checked"));
  const rows = el("ol");
  trace.append(rows);
  li.append(el("p", "q", question), answer, chips, trace);
  $("#chat").append(li);
  let items = [];
  const open = (i) => {
    const p = items[i];
    if (!p) return;
    li.querySelector(".passage")?.remove();
    const box = el("div", "passage");
    box.append(el("span", "meta", [p.section, p.page != null ? "page " + p.page : ""].filter(Boolean).join(" · ") || "passage " + (i + 1)), p.text);
    chips.after(box);
  };
  const scroll = () => li.scrollIntoView({ block: "end", behavior: "smooth" });
  return {
    event(ev) {
      if (ev.type === "step") rows.append(el("li", "step", ev.text));
      else if (ev.type === "gate") {
        const row = el("li", "gate", `${ev.gate}: ${ev.n === 1 ? ev.labels[0] : ev.labels.filter((l) => l === "yes").length + " of " + ev.n + " yes"}`);
        row.append(el("span", "badge " + (ev.who.includes("Gemma") ? "gemma" : "laya"), ev.who), el("span", "ms", ev.ms + " ms"));
        rows.append(row);
      } else if (ev.type === "passages") {
        items = ev.items;
        items.forEach((_, i) => { const b = el("button", "", `[${i + 1}]`); b.type = "button"; b.setAttribute("aria-label", "Source " + (i + 1)); b.onclick = () => open(i); chips.append(b); });
      } else if (ev.type === "answer") {
        answer.className = "a" + (ev.refused ? " refused" : "");
        if (!ev.refused) renderAnswer(answer, ev.text, open);
        else answer.textContent = ev.reason === "off_topic" ? "Not about machine learning, so I didn't look it up."
          : `No passage in the ${label.toLowerCase()} answered this (query rewritten twice).`;
      } else if (ev.type === "done") rows.append(el("li", "total", `total ${ev.seconds.toFixed(1)} s`));
      else if (ev.type === "error") this.error(ev.message);
      scroll();
    },
    error(msg) { answer.textContent = ""; answer.hidden = true; answer.classList.remove("pending"); li.append(el("p", "error", "Something went wrong: " + msg)); scroll(); },
  };
}

async function send(question) {
  question = question.trim();
  if (!question || busy || !picked("corpus")) return;
  busy = true; $("#send").disabled = true; $("#welcome").hidden = true; $("#question").value = "";
  const label = corpora.find((c) => c.name === picked("corpus")).label;
  const turn = newTurn(question, label);
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: picked("mode"), corpus: picked("corpus") }) });
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
    turn.error(String(e));
  } finally {
    busy = false; $("#send").disabled = !picked("corpus"); $("#question").focus();
  }
}

$("#ask").addEventListener("submit", (e) => { e.preventDefault(); send($("#question").value); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($("#question").value); } });
$("#new-chat").addEventListener("click", () => { if (!busy) { $("#chat").replaceChildren(); $("#welcome").hidden = false; } });
load();