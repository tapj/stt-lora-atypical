window.addEventListener("DOMContentLoaded", () => {
  let mediaRecorder = null;
  let chunks = [];
  let lastTranscript = "";
  let currentBlob = null;
  let currentFile = null;

  const recordBtn = document.getElementById("recordBtn");
  const stopBtn = document.getElementById("stopBtn");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadFile = document.getElementById("uploadFile");
  const transcribeBtn = document.getElementById("transcribeBtn");
  const copyBtn = document.getElementById("copyBtn");
  const saveBtn = document.getElementById("saveBtn");
  const statusEl = document.getElementById("status");
  const outputEl = document.getElementById("output");
  const downloadModelBtn = document.getElementById("downloadModelBtn");
  const modelSelect = document.getElementById("modelSelect");
  const customModelInput = document.getElementById("customModel");
  const enableLora = document.getElementById("enableLora");
  const loraPanel = document.getElementById("loraPanel");
  const loraSelect = document.getElementById("loraSelect");
  const refreshLoraBtn = document.getElementById("refreshLoraBtn");
  const uploadZipBtn = document.getElementById("uploadZipBtn");
  const zipFile = document.getElementById("zipFile");
  const datasetNameInput = document.getElementById("datasetName");
  const runNameInput = document.getElementById("runName");
  const trainBtn = document.getElementById("trainBtn");
  const trainLog = document.getElementById("trainLog");

  console.log("UI elements", { recordBtn, transcribeBtn, uploadBtn, uploadFile });

  if (!recordBtn || !transcribeBtn || !uploadBtn || !uploadFile) {
    console.error("Missing DOM elements. Check ids in index.html.");
    return;
  }

  function setStatus(s) {
    statusEl.textContent = s;
  }

  function setTranscript(text, timestamp) {
    lastTranscript = timestamp ? `[${timestamp}] ${text}` : text;
    outputEl.value = lastTranscript;
    copyBtn.disabled = false;
    saveBtn.disabled = false;
  }

  async function readJsonOrText(resp) {
    const ct = resp.headers.get("content-type") || "";
    const raw = await resp.text();
    let data = null;
    if (ct.includes("application/json")) {
      try {
        data = JSON.parse(raw);
      } catch (e) {
        data = { error: raw || e.message };
      }
    } else {
      data = { error: raw };
    }
    return { data, raw };
  }

  async function fetchJsonSafe(url, options) {
    const resp = await fetch(url, options);
    const { data, raw } = await readJsonOrText(resp);
    if (!resp.ok) throw new Error(data.error || data.detail || raw || resp.statusText);
    return data;
  }

  function currentAdapterDir() {
    if (enableLora && enableLora.checked) {
      const val = loraSelect && loraSelect.value;
      if (val) {
        document.getElementById("adapterDir").value = val;
        return val;
      }
      return "";
    }
    return "";
  }

  function buildFormData(file) {
    const fd = new FormData();
    fd.append("audio", file, file.name || "audio.wav");
    fd.append("adapter_dir", currentAdapterDir());
    fd.append("device", document.getElementById("device").value);
    fd.append("language", document.getElementById("language").value);
    fd.append("beam_size", document.getElementById("beam").value);
    fd.append("temperature", document.getElementById("temp").value);
    return fd;
  }

  async function transcribeFile(file, message) {
    console.log("transcribe start", file && file.name);
    transcribeBtn.disabled = true;
    uploadBtn.disabled = true;
    setStatus(message);

    try {
      const data = await fetchJsonSafe("/api/transcribe", { method: "POST", body: buildFormData(file) });
      setTranscript(data.text || "");
      setStatus("Done.");
    } catch (err) {
      setStatus("Error: " + err.message);
    } finally {
      transcribeBtn.disabled = false;
      uploadBtn.disabled = false;
    }
  }

  recordBtn.onclick = async () => {
    chunks = [];
    lastTranscript = "";
    outputEl.value = "";

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      currentBlob = new Blob(chunks, { type: "audio/webm" });
      currentFile = null;
      setStatus("Recorded. Ready to transcribe.");
      console.log("recording stopped", currentBlob);
    };

    mediaRecorder.start();
    setStatus("Recording...");
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    transcribeBtn.disabled = true;
    copyBtn.disabled = true;
    saveBtn.disabled = true;
  };

  stopBtn.onclick = () => {
    if (!mediaRecorder) return;
    mediaRecorder.stop();
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    transcribeBtn.disabled = false;
  };

  uploadBtn.onclick = () => {
    currentFile = null;
    uploadFile.value = "";
    uploadFile.click();
  };

  uploadFile.onchange = async () => {
    const f = uploadFile.files && uploadFile.files[0];
    if (!f) return;
    currentFile = f;
    currentBlob = null;
    copyBtn.disabled = true;
    saveBtn.disabled = true;
    setStatus(`Selected: ${f.name}`);
    console.log("file selected", f.name);
  };

  transcribeBtn.onclick = async (e) => {
    e.preventDefault();
    console.log("transcribe click");

    let fileToSend = null;
    if (currentFile) {
      fileToSend = currentFile;
    } else if (currentBlob) {
      const mimeType = currentBlob.type || "audio/webm";
      fileToSend = new File([currentBlob], "recording.webm", { type: mimeType });
    }

    if (!fileToSend) {
      setStatus("No audio selected or recorded.");
      return;
    }

    await transcribeFile(fileToSend, "Transcribing...");
  };

  copyBtn.onclick = async () => {
    await navigator.clipboard.writeText(outputEl.value);
    setStatus("Copied.");
  };

  saveBtn.onclick = () => {
    const text = outputEl.value;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Saved.");
  };

  downloadModelBtn.onclick = async () => {
    const custom = customModelInput.value.trim();
    const modelId = custom || modelSelect.value;
    if (!modelId) {
      setStatus("Pick a model or enter a custom repo id.");
      return;
    }

    setStatus(`Downloading ${modelId}...`);
    downloadModelBtn.disabled = true;

    const fd = new FormData();
    fd.append("model_id", modelId);

    try {
      const data = await fetchJsonSafe("/api/download_model", { method: "POST", body: fd });
      setStatus(`Downloaded ${data.model_id} to HF cache.`);
    } catch (err) {
      setStatus("Error: " + err.message);
    }

    downloadModelBtn.disabled = false;
  };

  async function refreshLoraList() {
    if (!enableLora.checked) return;
    try {
      const data = await fetchJsonSafe("/api/lora/list");
      loraSelect.innerHTML = "";
      data.lora.forEach((item) => {
        const opt = document.createElement("option");
        opt.value = item.path;
        opt.textContent = item.name;
        loraSelect.appendChild(opt);
      });
      if (loraSelect.options.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "(aucun adapter)";
        loraSelect.appendChild(opt);
      }
    } catch (err) {
      setStatus("Erreur liste LoRA: " + err.message);
    }
  }

  function toggleLoraPanel() {
    if (!enableLora) return;
    const show = enableLora.checked;
    loraPanel.style.display = show ? "block" : "none";
    if (show) {
      refreshLoraList();
    } else {
      loraSelect.value = "";
      document.getElementById("adapterDir").value = "";
    }
  }

  enableLora && enableLora.addEventListener("change", toggleLoraPanel);
  refreshLoraBtn && refreshLoraBtn.addEventListener("click", refreshLoraList);

  uploadZipBtn &&
    uploadZipBtn.addEventListener("click", () => {
      if (zipFile) zipFile.value = "";
      zipFile && zipFile.click();
    });

  zipFile &&
    (zipFile.onchange = async () => {
      const file = zipFile.files && zipFile.files[0];
      const datasetName = (datasetNameInput.value || "").trim();
      if (!file || !datasetName) {
        setStatus("Choisir un fichier ZIP et dataset_name.");
        return;
      }

      const fd = new FormData();
      fd.append("zipfile", file);
      fd.append("dataset_name", datasetName);

      try {
        const data = await fetchJsonSafe("/api/lora/upload_zip", { method: "POST", body: fd });
        setStatus(`Dataset ${data.dataset_dir} importé (${data.n_pairs} paires).`);
      } catch (err) {
        setStatus("Erreur upload ZIP: " + err.message);
      }
    });

  let statusInterval = null;

  function stopStatusPolling() {
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }
  }

  async function pollStatus(runName) {
    try {
      const data = await fetchJsonSafe(`/api/lora/status?run_name=${encodeURIComponent(runName)}`);
      const statusObj = data.status || {};
      const lines = data.log_tail || [];
      trainLog.textContent = `Etat: ${statusObj.state || "?"}\n${lines.join("")}`;
      if (statusObj.state === "done" || statusObj.state === "error") {
        stopStatusPolling();
        refreshLoraList();
      }
    } catch (err) {
      trainLog.textContent = "Erreur statut: " + err.message;
      stopStatusPolling();
    }
  }

  trainBtn &&
    trainBtn.addEventListener("click", async () => {
      const datasetName = (datasetNameInput.value || "").trim();
      const runName = (runNameInput.value || "").trim();
      if (!datasetName || !runName) {
        setStatus("dataset_name et run_name requis.");
        return;
      }

      const fd = new FormData();
      fd.append("dataset_name", datasetName);
      fd.append("run_name", runName);
      fd.append("base_model", "openai/whisper-small");
      fd.append("epochs", "3");
      fd.append("lr", "1e-4");
      fd.append("batch_size", "8");

      try {
        const data = await fetchJsonSafe("/api/lora/train", { method: "POST", body: fd });
        setStatus(`Entraînement lancé: ${data.run_name}`);
        trainLog.textContent = "Démarrage de l'entraînement...\n";
        stopStatusPolling();
        statusInterval = setInterval(() => pollStatus(runName), 3000);
      } catch (err) {
        setStatus("Erreur entraînement: " + err.message);
      }
    });

  toggleLoraPanel();
});
