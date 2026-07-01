const state = {
  health: null,
  status: null,
};

const elements = {
  refreshStatus: document.querySelector("#refresh-status"),
  statusBadge: document.querySelector("#status-badge"),
  apiStatus: document.querySelector("#api-status"),
  weaviateStatus: document.querySelector("#weaviate-status"),
  collectionName: document.querySelector("#collection-name"),
  generatorModel: document.querySelector("#generator-model"),
  statusMessage: document.querySelector("#status-message"),
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
  answerOutput: document.querySelector("#answer-output"),
  sourcesList: document.querySelector("#sources-list"),
};

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
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

async function refreshDocuments() {
  elements.documentsList.innerHTML = '<p class="message muted">Loading documents...</p>';
  try {
    const data = await requestJson("/rag/documents");
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
    metaSpan(`type: ${documentSummary.doc_type || "unknown"}`),
    metaSpan(`source_id: ${documentSummary.source_id}`),
    metaSpan(`chunks: ${documentSummary.chunk_count}`),
  );
  if (documentSummary.page_numbers && documentSummary.page_numbers.length) {
    meta.append(metaSpan(`pages: ${documentSummary.page_numbers.join(", ")}`));
  }
  if (documentSummary.document_hash) {
    meta.append(metaSpan(`document_hash: ${documentSummary.document_hash}`));
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
    downloadButton.addEventListener("click", () => {
      window.open(
        `/rag/documents/${encodeURIComponent(documentSummary.source_id)}/download`,
        "_blank",
        "noopener",
      );
    });
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

async function deleteDocument(sourceId) {
  const confirmed = window.confirm(
    `Delete indexed chunks and any stored original file for source_id ${sourceId}?`,
  );
  if (!confirmed) {
    return;
  }

  try {
    const result = await requestJson(`/rag/documents/${encodeURIComponent(sourceId)}`, {
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
  };

  elements.answerOutput.textContent = "Asking...";
  elements.answerOutput.className = "message muted";
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
    renderSources(result.sources || []);
  } catch (error) {
    elements.answerOutput.textContent = error.message;
    elements.answerOutput.className = "message error-text";
  } finally {
    setFormBusy(elements.queryForm, false);
  }
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
    metaSpan(`type: ${source.doc_type || "unknown"}`),
    metaSpan(`chunk: ${source.chunk_index ?? "unknown"}`),
    metaSpan(`page: ${source.page_number ?? "unknown"}`),
    metaSpan(`distance: ${formatDistance(source.distance)}`),
  );

  const preview = document.createElement("p");
  preview.className = "preview";
  preview.textContent = source.text || "";

  article.append(title, meta, preview);
  return article;
}

function formatDistance(distance) {
  if (distance === null || distance === undefined) {
    return "unknown";
  }
  return Number(distance).toFixed(4);
}

function setFormBusy(form, busy) {
  form.querySelectorAll("button, input, textarea").forEach((control) => {
    control.disabled = busy;
  });
}

elements.refreshStatus.addEventListener("click", refreshStatus);
elements.refreshDocuments.addEventListener("click", refreshDocuments);
elements.fileForm.addEventListener("submit", ingestFile);
elements.textForm.addEventListener("submit", ingestText);
elements.queryForm.addEventListener("submit", askQuestion);

refreshStatus();
refreshDocuments();
