/* Braid demo front end. No framework, no build step, no dependencies.
 *
 * Streaming uses fetch + a manual SSE reader rather than EventSource, because
 * the completion endpoint is a POST and EventSource cannot send a body.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
	prompt: $("prompt"),
	maxNewBytes: $("max-new-bytes"),
	temperature: $("temperature"),
	topK: $("top-k"),
	seed: $("seed"),
	generate: $("generate"),
	stop: $("stop"),
	clear: $("clear"),
	output: $("output"),
	stats: $("stats"),
	statusText: $("status-text"),
	statusDot: $("status-dot"),
	meta: $("model-meta"),
	limitations: $("limitations"),
	notice: $("backend-notice"),
	examples: $("examples"),
	repoLink: $("repo-link"),
	siteLink: $("site-link"),
};

let controller = null;
let startedAt = 0;
let producedBytes = 0;
let tickTimer = null;

const encoder = new TextEncoder();

/* ------------------------------------------------------------- utilities */

function setStatus(text, kind) {
	els.statusText.textContent = text;
	els.statusDot.className = "dot" + (kind ? " " + kind : "");
}

function formatNumber(n) {
	if (n === null || n === undefined) return "—";
	return Number(n).toLocaleString();
}

function formatBytes(n) {
	if (!n && n !== 0) return "—";
	if (n < 1024) return n + " B";
	if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KiB";
	return (n / 1048576).toFixed(1) + " MiB";
}

function showNotice(message, isError) {
	if (!message) {
		els.notice.hidden = true;
		return;
	}
	els.notice.hidden = false;
	els.notice.textContent = message;
	els.notice.className = "notice" + (isError ? " error" : "");
}

function updateStats(final) {
	const elapsed = (performance.now() - startedAt) / 1000;
	const rate = elapsed > 0 ? producedBytes / elapsed : 0;
	els.stats.textContent =
		`${producedBytes} B · ${elapsed.toFixed(2)} s · ${rate.toFixed(0)} B/s` +
		(final ? "" : " …");
}

/* ------------------------------------------------------------ model info */

async function loadModel() {
	try {
		const response = await fetch("/v1/model");
		if (!response.ok) {
			const body = await response.json().catch(() => ({}));
			throw new Error(body.detail || body.error || `HTTP ${response.status}`);
		}
		const info = await response.json();
		renderModel(info);
		if (info.backend === "mock") {
			showNotice(
				"This deployment is running the MOCK backend. Output is deterministic " +
				"placeholder text, not model output.",
				false
			);
		}
	} catch (err) {
		els.meta.innerHTML = "<dt>status</dt><dd>unavailable</dd>";
		showNotice("The model is not available right now: " + err.message, true);
		setStatus("model unavailable", "error");
		els.generate.disabled = true;
	}
}

function renderModel(info) {
	const rows = [
		["name", info.model_name],
		["backend", info.backend],
		["parameters", info.parameters ? formatNumber(info.parameters) : "—"],
		["architecture", info.architecture],
		["details", info.architecture_summary],
		["context limit", formatBytes(info.context_limit_bytes)],
		["max prompt", formatBytes(info.max_prompt_bytes)],
		["max output", formatBytes(info.max_new_bytes)],
		["checkpoint", info.checkpoint_version || "—"],
		["checkpoint id", info.checkpoint_id || "—"],
		["runtime", info.runtime_version],
		["device", info.device + (info.precision ? " / " + info.precision : "")],
		["preset", info.preset || "—"],
		["training bytes", info.training_bytes ? formatNumber(info.training_bytes) : "not published"],
		["dataset", info.dataset || "not published"],
		["licence", info.license || "not published"],
		["status", info.research_preview ? "research preview" : "—"],
	];
	els.meta.innerHTML = rows
		.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v ?? "—"))}</dd>`)
		.join("");

	if (Array.isArray(info.known_limitations) && info.known_limitations.length) {
		els.limitations.innerHTML = info.known_limitations
			.map((item) => `<li>${escapeHtml(item)}</li>`)
			.join("");
	}
	if (info.max_new_bytes) {
		els.maxNewBytes.max = String(info.max_new_bytes);
		if (Number(els.maxNewBytes.value) > info.max_new_bytes) {
			els.maxNewBytes.value = String(info.max_new_bytes);
			syncRangeLabels();
		}
	}
}

function escapeHtml(text) {
	return text.replace(/[&<>"']/g, (c) => ({
		"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
	}[c]));
}

/* ------------------------------------------------------------ generation */

function requestBody() {
	const body = {
		prompt: els.prompt.value,
		max_new_bytes: Number(els.maxNewBytes.value),
		temperature: Number(els.temperature.value),
		top_k: Number(els.topK.value),
	};
	const seed = els.seed.value.trim();
	if (seed !== "") body.seed = Number(seed);
	return body;
}

async function generate() {
	if (controller) return;
	controller = new AbortController();
	els.generate.disabled = true;
	els.stop.disabled = false;
	els.output.textContent = "";
	producedBytes = 0;
	startedAt = performance.now();
	setStatus("generating", "live");
	showNotice(null);
	tickTimer = setInterval(() => updateStats(false), 200);

	try {
		const response = await fetch("/v1/completions/stream", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(requestBody()),
			signal: controller.signal,
		});
		if (!response.ok) {
			const body = await response.json().catch(() => ({}));
			throw new Error(body.detail || body.error || `HTTP ${response.status}`);
		}
		await readEventStream(response.body);
		setStatus("done");
	} catch (err) {
		if (err.name === "AbortError") {
			setStatus("stopped");
		} else {
			setStatus("error", "error");
			showNotice("Generation failed: " + err.message, true);
		}
	} finally {
		clearInterval(tickTimer);
		updateStats(true);
		controller = null;
		els.generate.disabled = false;
		els.stop.disabled = true;
	}
}

async function readEventStream(stream) {
	const reader = stream.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let split;
		while ((split = buffer.indexOf("\n\n")) !== -1) {
			const frame = buffer.slice(0, split);
			buffer = buffer.slice(split + 2);
			handleFrame(frame);
		}
	}
	if (buffer.trim()) handleFrame(buffer);
}

function handleFrame(frame) {
	let event = "message";
	const dataLines = [];
	for (const line of frame.split("\n")) {
		if (line.startsWith("event:")) event = line.slice(6).trim();
		else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
	}
	if (!dataLines.length) return;
	let payload;
	try {
		payload = JSON.parse(dataLines.join("\n"));
	} catch {
		return;
	}
	if (event === "chunk" && typeof payload.text === "string") {
		els.output.textContent += payload.text;
		producedBytes += encoder.encode(payload.text).length;
		els.output.scrollTop = els.output.scrollHeight;
	} else if (event === "start" && payload.truncated_prompt) {
		showNotice("The prompt was longer than the server limit and was truncated.", false);
	} else if (event === "error") {
		throwLater(payload.detail || payload.error || "generation failed");
	} else if (event === "done") {
		producedBytes = payload.generated_bytes ?? producedBytes;
	}
}

function throwLater(message) {
	setStatus("error", "error");
	showNotice("Generation failed: " + message, true);
}

function stop() {
	if (controller) controller.abort();
}

/* ------------------------------------------------------------------ wire */

function syncRangeLabels() {
	$("max-new-bytes-value").textContent = els.maxNewBytes.value;
	$("temperature-value").textContent = Number(els.temperature.value).toFixed(2);
	$("top-k-value").textContent = els.topK.value === "0" ? "off" : els.topK.value;
}

els.maxNewBytes.addEventListener("input", syncRangeLabels);
els.temperature.addEventListener("input", syncRangeLabels);
els.topK.addEventListener("input", syncRangeLabels);
els.generate.addEventListener("click", generate);
els.stop.addEventListener("click", stop);
els.clear.addEventListener("click", () => {
	els.output.textContent = "";
	els.stats.textContent = "";
	setStatus("idle");
});
els.examples.addEventListener("click", (event) => {
	if (event.target.classList.contains("chip")) {
		els.prompt.value = event.target.textContent;
		els.prompt.focus();
	}
});
els.prompt.addEventListener("keydown", (event) => {
	if ((event.metaKey || event.ctrlKey) && event.key === "Enter") generate();
});

syncRangeLabels();
setStatus("idle");
loadModel();
