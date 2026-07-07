const state = {
  health: null,
  status: null,
  apiKey: localStorage.getItem("rag_api_key") || "",
  projectId: localStorage.getItem("rag_project_id") || "",
};

const elements = {
  refreshStatus: document.querySelector("#refresh-status"),
  statusBadge: document.querySelector("#status-badge"),
  apiStatus: document.querySelector("#api-status"),
  weaviateStatus: document.querySelector("#weaviate-status"),
  collectionName: document.querySelector("#collection-name"),
  generatorModel: document.querySelector("#generator-model"),
  statusMessage: document.querySelector("#status-message"),
  uploadLimits: document.querySelector("#upload-limits"),
  settingsForm: document.querySelector("#settings-form"),
  apiKeyInput: document.querySelector("#api-key-input"),
  projectIdInput: document.querySelector("#project-id-input"),
  currentProject: document.querySelector("#current-project"),
  clearSettings: document.querySelector("#clear-settings"),
  fileForm: document.querySelector("#file-form"),
  fileInput: document.querySelector("#file-input"),
  fileDocType: document.querySelector("#file-doc-type"),
  fileResult: document.querySelector("#file-result"),
  textForm: document.querySelector("#text-form"),
  textInput: document.querySelector("#text-input"),
  textFilename: document.querySelector("#text-filename"),
  textDocType: document.querySelector("#text-doc-type"),
  textResult: document.querySelector("#text-result"),
  refreshDocuments: document.querySelector("#refresh-documents"),
  documentsList: document.querySelector("#documents-list"),
  queryForm: document.querySelector("#query-form"),
  questionInput: document.querySelector("#question-input"),
  topKInput: document.querySelector("#top-k-input"),
  retrievalModeInput: document.querySelector("#retrieval-mode-input"),
  queryDocTypeInput: document.querySelector("#query-doc-type-input"),
  queryFilenameInput: document.querySelector("#query-filename-input"),
  querySourceIdInput: document.querySelector("#query-source-id-input"),
  answerMeta: document.querySelector("#answer-meta"),
  answerOutput: document.querySelector("#answer-output"),
  sourcesList: document.querySelector("#sources-list"),
};

async function requestJson(url, options = {}) {
  const preparedOptions = withAuthHeaders(options);
  let response;
  try {
    response = await fetch(url, preparedOptions);
  } catch (error) {
    throw new Error(`API unavailable: ${error.message}`);
  }

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(detail || `Request failed with HTTP ${response.status}`);
  }
  return body;
}

function withAuthHeaders(options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) {
    headers.set("X-RAG-API-Key", state.apiKey);
  }
  return { ...options, headers };
}

function currentProjectId() {
  const fallback = state.status?.default_project_id || "default";
  return (state.projectId || fallback).trim() || fallback;
}

function applyProjectToPayload(payload) {
  payload.project_id = currentProjectId();
}

function projectQueryString() {
  return `project_id=${encodeURIComponent(currentProjectId())}`;
}

function renderJson(target, data) {
  target.textContent = JSON.stringify(data, null, 2);
  target.classList.remove("error-text");
}

function renderError(target, error) {
  target.textContent = error.message || String(error);
  target.classList.add("error-text");
}

async function refreshStatus() {
  elements.statusBadge.textContent = "Checking";
  elements.statusBadge.className = "badge";
  elements.statusMessage.textContent = "";

  try {
    const [health, status] = await Promise.all([
      requestJson("/health"),
      requestJson("/rag/status"),
    ]);
    state.health = health;
    state.status = status;
    elements.apiStatus.textContent = health.status || "ok";
    elements.weaviateStatus.textContent = status.weaviate_reachable
      ? "Reachable"
      : "Unavailable";
    elements.collectionName.textContent =
      status.weaviate_collection || health.weaviate_collection || "-";
    elements.generatorModel.textContent = health.generator_model || "-";
    elements.statusMessage.textContent = status.message || "";
    renderUploadLimits(health);
    renderCurrentProject();

    elements.statusBadge.textContent = status.weaviate_reachable ? "Ready" : "Degraded";
    elements.statusBadge.className = status.weaviate_reachable ? "badge ok" : "badge warn";
  } catch (error) {
    elements.apiStatus.textContent = "Unavailable";
    elements.weaviateStatus.textContent = "-";
    elements.collectionName.textContent = "-";
    elements.generatorModel.textContent = "-";
    elements.statusMessage.textContent = error.message;
    elements.statusBadge.textContent = "Offline";
    elements.statusBadge.className = "badge error";
  }
}

function initializeSettings() {
  elements.apiKeyInput.value = state.apiKey;
  elements.projectIdInput.value = state.projectId;
  renderCurrentProject();
}

function renderCurrentProject() {
  elements.currentProject.textContent = `Project: ${currentProjectId()}`;
}

function saveSettings(event) {
  event.preventDefault();
  state.apiKey = elements.apiKeyInput.value.trim();
  state.projectId = elements.projectIdInput.value.trim();
  if (state.apiKey) {
    localStorage.setItem("rag_api_key", state.apiKey);
  } else {
    localStorage.removeItem("rag_api_key");
  }
  if (state.projectId) {
    localStorage.setItem("rag_project_id", state.projectId);
  } else {
    localStorage.removeItem("rag_project_id");
  }
  renderCurrentProject();
  refreshDocuments();
}

function clearSettings() {
  state.apiKey = "";
  state.projectId = "";
  localStorage.removeItem("rag_api_key");
  localStorage.removeItem("rag_project_id");
  elements.apiKeyInput.value = "";
  elements.projectIdInput.value = "";
  renderCurrentProject();
  refreshDocuments();
}

function renderUploadLimits(health) {
  if (!elements.uploadLimits) {
    return;
  }
  const extensions = health.allowed_upload_extensions || [".pdf", ".txt", ".md", ".docx", ".csv"];
  const maxUploadMb = health.max_upload_mb || 25;
  elements.uploadLimits.textContent = `Allowed: ${extensions.join(", ")}. Max upload: ${maxUploadMb} MB.`;
}

async function refreshDocuments() {
  elements.documentsList.innerHTML = '<p class="message muted">Loading documents...</p>';
  try {
    const data = await requestJson(`/rag/documents?${projectQueryString()}`);
    const documents = data.documents || [];
    if (!documents.length) {
      elements.documentsList.innerHTML =
        '<p class="message muted">No indexed documents found.</p>';
      return;
    }
    elements.documentsList.replaceChildren(...documents.map(renderDocument));
  } catch (error) {
    elements.documentsList.innerHTML = "";
    const paragraph = document.createElement("p");
    paragraph.className = "message error-text";
    paragraph.textContent = error.message;
    elements.documentsList.append(paragraph);
  }
}

function renderDocument(documentSummary) {
  const article = document.createElement("article");
  article.className = "document-item";

  const wrapper = document.createElement("div");
  wrapper.className = "document-main";

  const body = document.createElement("div");
  const title = document.createElement("p");
  title.className = "item-title";
  title.textContent = documentSummary.filename || "Untitled document";

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.append(
    metaSpan(`project: ${documentSummary.project_id || currentProjectId()}`),
    metaSpan(`type: ${documentSummary.doc_type || "unknown"}`),
    metaSpan(`source_id: ${documentSummary.source_id}`),
    metaSpan(`chunks: ${documentSummary.chunk_count}`),
  );
  if (documentSummary.page_numbers && documentSummary.page_numbers.length) {
    meta.append(metaSpan(formatPageNumbers(documentSummary.page_numbers)));
  }
  if (documentSummary.document_hash) {
    meta.append(metaSpan(`document_hash: ${documentSummary.document_hash}`));
  }
  if (documentSummary.parser_used) {
    meta.append(metaSpan(`parser: ${documentSummary.parser_used}`));
  }
  if (documentSummary.detected_extension) {
    meta.append(metaSpan(`ext: ${documentSummary.detected_extension}`));
  }
  if (documentSummary.original_file_size_bytes !== null && documentSummary.original_file_size_bytes !== undefined) {
    meta.append(metaSpan(`size: ${formatBytes(documentSummary.original_file_size_bytes)}`));
  }
  const warnings = documentSummary.warnings || [];
  if (warnings.length) {
    meta.append(metaSpan(`warnings: ${warnings.length}`));
  }

  body.append(title, meta);
  const badges = document.createElement("div");
  badges.className = "badge-row";
  const fileBadge = document.createElement("span");
  fileBadge.className = documentSummary.original_file_available
    ? "badge ok"
    : "badge warn";
  fileBadge.textContent = documentSummary.original_file_available
    ? "Original available"
    : "Original unavailable";
  badges.append(fileBadge);
  if (documentSummary.original_filename && documentSummary.original_filename !== documentSummary.filename) {
    badges.append(metaSpan(`original: ${documentSummary.original_filename}`));
  }
  body.append(badges);
  if (documentSummary.stored_file_path) {
    const storedPath = document.createElement("p");
    storedPath.className = "stored-path";
    storedPath.textContent = documentSummary.stored_file_path;
    body.append(storedPath);
  }
  if (documentSummary.preview) {
    const preview = document.createElement("p");
    preview.className = "preview";
    preview.textContent = documentSummary.preview;
    body.append(preview);
  }

  const actions = document.createElement("div");
  actions.className = "document-actions";
  if (documentSummary.original_file_available) {
    const downloadButton = document.createElement("button");
    downloadButton.type = "button";
    downloadButton.className = "secondary";
    downloadButton.textContent = "Download";
    downloadButton.addEventListener("click", () => downloadDocument(documentSummary));
    actions.append(downloadButton);
  }
  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "danger";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", () => deleteDocument(documentSummary.source_id));
  actions.append(deleteButton);

  wrapper.append(body, actions);
  article.append(wrapper);
  return article;
}

function metaSpan(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

async function downloadDocument(documentSummary) {
  const url = `/rag/documents/${encodeURIComponent(documentSummary.source_id)}/download?${projectQueryString()}`;
  try {
    const response = await fetch(url, withAuthHeaders());
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed with HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = documentSummary.filename || "document";
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (error) {
    window.alert(`Download failed: ${error.message}`);
  }
}

async function deleteDocument(sourceId) {
  const confirmed = window.confirm(
    `Delete indexed chunks and any stored original file for source_id ${sourceId}?`,
  );
  if (!confirmed) {
    return;
  }

  try {
    const result = await requestJson(`/rag/documents/${encodeURIComponent(sourceId)}?${projectQueryString()}`, {
      method: "DELETE",
    });
    await refreshDocuments();
    const deletedFiles = result.deleted_files || [];
    elements.statusMessage.textContent = deletedFiles.length
      ? `${result.message} Removed: ${deletedFiles.join(", ")}`
      : `${result.message} No stored original file was removed.`;
  } catch (error) {
    window.alert(`Delete failed: ${error.message}`);
  }
}

async function ingestFile(event) {
  event.preventDefault();
  const file = elements.fileInput.files[0];
  if (!file) {
    renderError(elements.fileResult, new Error("Choose a PDF or TXT file first."));
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("doc_type", elements.fileDocType.value || "general");
  formData.append("project_id", currentProjectId());

  setFormBusy(elements.fileForm, true);
  try {
    const result = await requestJson("/rag/ingest/file", {
      method: "POST",
      body: formData,
    });
    renderJson(elements.fileResult, result);
    await refreshDocuments();
  } catch (error) {
    renderError(elements.fileResult, error);
  } finally {
    setFormBusy(elements.fileForm, false);
  }
}

async function ingestText(event) {
  event.preventDefault();
  const payload = {
    text: elements.textInput.value,
    filename: elements.textFilename.value || "manual-note.txt",
    doc_type: elements.textDocType.value || "general",
  };
  applyProjectToPayload(payload);

  setFormBusy(elements.textForm, true);
  try {
    const result = await requestJson("/rag/ingest/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderJson(elements.textResult, result);
    await refreshDocuments();
  } catch (error) {
    renderError(elements.textResult, error);
  } finally {
    setFormBusy(elements.textForm, false);
  }
}

async function askQuestion(event) {
  event.preventDefault();
  const payload = {
    question: elements.questionInput.value,
    top_k: Number(elements.topKInput.value || 3),
    retrieval_mode: elements.retrievalModeInput.value || "vector",
  };
  applyProjectToPayload(payload);
  addOptionalFilter(payload, "doc_type", elements.queryDocTypeInput.value);
  addOptionalFilter(payload, "filename", elements.queryFilenameInput.value);
  addOptionalFilter(payload, "source_id", elements.querySourceIdInput.value);

  elements.answerOutput.textContent = "Asking...";
  elements.answerOutput.className = "message muted";
  elements.answerMeta.textContent = "";
  elements.sourcesList.innerHTML = "";
  setFormBusy(elements.queryForm, true);

  try {
    const result = await requestJson("/rag/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    elements.answerOutput.textContent = result.answer || "No answer returned.";
    elements.answerOutput.className = "message";
    renderAnswerMeta(result);
    renderSources(result.sources || []);
  } catch (error) {
    elements.answerOutput.textContent = error.message;
    elements.answerOutput.className = "message error-text";
    elements.answerMeta.textContent = "";
  } finally {
    setFormBusy(elements.queryForm, false);
  }
}

function addOptionalFilter(payload, field, value) {
  const normalized = (value || "").trim();
  if (normalized) {
    payload[field] = normalized;
  }
}

function renderAnswerMeta(result) {
  const parts = [];
  if (result.retrieval_mode) {
    parts.push(`mode: ${result.retrieval_mode}`);
  }
  const filters = result.filters_applied || {};
  const filterEntries = Object.entries(filters).map(([key, value]) => `${key}=${value}`);
  if (filterEntries.length) {
    parts.push(`filters: ${filterEntries.join(", ")}`);
  }
  elements.answerMeta.textContent = parts.join(" | ");
}

function renderSources(sources) {
  if (!sources.length) {
    elements.sourcesList.innerHTML =
      '<p class="message muted">No sources returned.</p>';
    return;
  }
  elements.sourcesList.replaceChildren(...sources.map(renderSource));
}

function renderSource(source) {
  const article = document.createElement("article");
  article.className = "source-item";

  const title = document.createElement("p");
  title.className = "item-title";
  title.textContent = source.filename || "Unknown source";

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.append(
    metaSpan(`project: ${source.project_id || currentProjectId()}`),
    metaSpan(`type: ${source.doc_type || "unknown"}`),
    metaSpan(`source_id: ${source.source_id || "unknown"}`),
    metaSpan(`chunk: ${source.chunk_index ?? "unknown"}`),
    metaSpan(`page: ${source.page_number ?? "unknown"}`),
    metaSpan(formatSourceScore(source)),
  );
  if (source.section_title) {
    meta.append(metaSpan(`section: ${source.section_title}`));
  }
  if (source.row_number !== null && source.row_number !== undefined) {
    meta.append(metaSpan(`row: ${source.row_number}`));
  }
  if (source.chunk_char_count !== null && source.chunk_char_count !== undefined) {
    meta.append(metaSpan(`chars: ${source.chunk_char_count}`));
  }

  const preview = document.createElement("p");
  preview.className = "preview";
  preview.textContent = source.text || "";

  article.append(title, meta, preview);
  return article;
}

function formatPageNumbers(pageNumbers) {
  const visiblePages = pageNumbers.slice(0, 12);
  const remainder = pageNumbers.length - visiblePages.length;
  const suffix = remainder > 0 ? `, ... +${remainder} more` : "";
  return `pages: ${visiblePages.join(", ")}${suffix}`;
}

function formatSourceScore(source) {
  if (source.retrieval_score !== null && source.retrieval_score !== undefined) {
    return `score: ${Number(source.retrieval_score).toFixed(4)}`;
  }
  return `distance: ${formatDistance(source.distance)}`;
}

function formatDistance(distance) {
  if (distance === null || distance === undefined) {
    return "unknown";
  }
  return Number(distance).toFixed(4);
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value)) {
    return "unknown";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  const kb = value / 1024;
  if (kb < 1024) {
    return `${kb.toFixed(1)} KB`;
  }
  return `${(kb / 1024).toFixed(1)} MB`;
}

function setFormBusy(form, busy) {
  form.querySelectorAll("button, input, select, textarea").forEach((control) => {
    control.disabled = busy;
  });
}

elements.refreshStatus.addEventListener("click", refreshStatus);
elements.refreshDocuments.addEventListener("click", refreshDocuments);
elements.settingsForm.addEventListener("submit", saveSettings);
elements.clearSettings.addEventListener("click", clearSettings);
elements.fileForm.addEventListener("submit", ingestFile);
elements.textForm.addEventListener("submit", ingestText);
elements.queryForm.addEventListener("submit", askQuestion);

initializeSettings();
refreshStatus();
refreshDocuments();
