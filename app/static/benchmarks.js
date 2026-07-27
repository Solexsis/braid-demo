/* Renders /v1/benchmarks. Displays only what the data file contains; a null
 * value renders as "not measured", never as a zero or a guess. */
"use strict";

const container = document.getElementById("sections");
const provenance = document.getElementById("provenance");
const caveatsBox = document.getElementById("global-caveats");

function escapeHtml(text) {
	return String(text).replace(/[&<>"']/g, (c) => ({
		"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
	}[c]));
}

function formatCell(value, column, row) {
	if (value === null || value === undefined || value === "") {
		return '<span class="badge pending">not measured</span>';
	}
	switch (column.kind) {
		case "int":
			return Number(value).toLocaleString();
		case "float": {
			const ci = column.ci ? row[column.ci] : null;
			const base = Number(value).toFixed(value < 10 ? 4 : 1);
			return ci ? `${base} <span style="color:var(--fg-faint)">±${Number(ci).toFixed(4)}</span>` : base;
		}
		case "signed": {
			const n = Number(value);
			const sign = n > 0 ? "+" : "";
			const color = n > 0 ? "var(--danger)" : (n < 0 ? "var(--accent)" : "var(--fg-dim)");
			return `<span style="color:${color}">${sign}${n.toFixed(2)}%</span>`;
		}
		default:
			return escapeHtml(value);
	}
}

function renderTableSection(section) {
	const columns = section.columns || [];
	const head = columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("");
	const body = (section.rows || []).map((row) => {
		const cells = columns.map((column) => {
			const numeric = column.kind !== "text";
			return `<td class="${numeric ? "num" : ""}">${formatCell(row[column.key], column, row)}</td>`;
		}).join("");
		return `<tr class="${row.highlight ? "highlight" : ""}">${cells}</tr>`;
	}).join("");
	return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderContextSection(section) {
	const contexts = new Set();
	for (const row of section.rows || []) {
		for (const point of row.points || []) contexts.add(point.context_bytes);
	}
	const ordered = [...contexts].sort((a, b) => a - b);
	const head = ["architecture", ...ordered.map((c) => `${c} B`)]
		.map((label) => `<th>${escapeHtml(label)}</th>`).join("");
	const body = (section.rows || []).map((row) => {
		const byContext = new Map((row.points || []).map((p) => [p.context_bytes, p]));
		const cells = ordered.map((context) => {
			const point = byContext.get(context);
			return `<td class="num">${point
				? Number(point.train_bytes_per_second).toLocaleString()
				: '<span class="badge pending">n/a</span>'}</td>`;
		}).join("");
		return `<tr class="${row.kind === "braid" ? "highlight" : ""}"><td>${escapeHtml(row.label)}</td>${cells}</tr>`;
	}).join("");
	return `<p style="color:var(--fg-dim);font-size:0.86rem;margin:0 0 10px">Training throughput in bytes per second (higher is better).</p>
		<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderPendingSection(section) {
	const rows = (section.rows || []).map((row) => `
		<tr>
			<td>${escapeHtml(row.label)}</td>
			<td class="num">${Number(row.parameters).toLocaleString()}</td>
			<td class="num">${Number(row.context_limit_bytes).toLocaleString()} B</td>
			<td class="num"><span class="badge pending">awaiting checkpoint</span></td>
			<td class="num"><span class="badge pending">awaiting checkpoint</span></td>
		</tr>`).join("");
	return `<div class="table-scroll"><table><thead><tr>
			<th>preset</th><th>parameters (exact)</th><th>context limit</th>
			<th>val bits/byte</th><th>throughput</th>
		</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderSection(section) {
	const badge = section.status === "measured"
		? '<span class="badge measured">measured</span>'
		: '<span class="badge pending">pending</span>';
	let table;
	if (section.id === "context-scaling") table = renderContextSection(section);
	else if (section.status === "pending") table = renderPendingSection(section);
	else table = renderTableSection(section);

	const caveats = (section.caveats || [])
		.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

	return `
		<section class="panel" style="margin-top:20px">
			<h2 style="display:flex;gap:10px;align-items:center">
				<span>${escapeHtml(section.title)}</span> ${badge}
			</h2>
			${section.hardware ? `<p style="color:var(--fg-dim);font-size:0.84rem;margin:0 0 12px"><strong>Hardware:</strong> ${escapeHtml(section.hardware)}</p>` : ""}
			${table}
			<h2 style="margin-top:18px">Methodology</h2>
			<p style="color:var(--fg-dim);font-size:0.86rem;margin:0">${escapeHtml(section.methodology || "")}</p>
			${caveats ? `<h2 style="margin-top:18px">Caveats</h2><ul class="plain">${caveats}</ul>` : ""}
		</section>`;
}

async function load() {
	try {
		const response = await fetch("/v1/benchmarks");
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const data = await response.json();

		const presets = (data.presets || [])
			.map((p) => `${p.name} = ${Number(p.parameters).toLocaleString()} params`)
			.join("\n");
		provenance.textContent =
			`generated ${data.generated_at}\n` +
			`by ${data.generated_by}\n` +
			`source commit ${data.source_commit || "unknown"}\n` +
			`sources: ${(data.sources || []).join(", ")}\n\n${presets}`;

		if ((data.global_caveats || []).length) {
			caveatsBox.hidden = false;
			caveatsBox.innerHTML = "<strong>Read this first.</strong><ul class='plain'>" +
				data.global_caveats.map((c) => `<li>${escapeHtml(c)}</li>`).join("") +
				"</ul>";
		}

		container.innerHTML = (data.sections || []).map(renderSection).join("");
	} catch (err) {
		container.innerHTML =
			`<section class="panel"><p>Benchmark data is unavailable: ${escapeHtml(err.message)}</p></section>`;
	}
}

load();
