window.addEventListener("DOMContentLoaded", () => {
  let mediaRecorder = null;
  let chunks = [];
  let lastTranscript = "";

  const recordBtn = document.getElementById("recordBtn");
  const stopBtn = document.getElementById("stopBtn");
  const uploadBtn = document.getElementById("uploadBtn");
  const wavFile = document.getElementById("wavFile");
  const transcribeBtn = document.getElementById("transcribeBtn");
  const copyBtn = document.getElementById("copyBtn");
  const saveBtn = document.getElementById("saveBtn");
  const statusEl = document.getElementById("status");
  const outputEl = document.getElementById("output");
  const downloadModelBtn = document.getElementById("downloadModelBtn");
  const modelSelect = document.getElementById("modelSelect");
  const customModelInput = document.getElementById("customModel");

  function setStatus(s) {
    statusEl.textContent = s;
  }

  function setTranscript(text, timestamp) {
    lastTranscript = timestamp ? `[${timestamp}] ${text}` : text;
    outputEl.value = lastTranscript;
    copyBtn.disabled = false;
    saveBtn.disabled = false;
  }

  function buildFormData(file) {
    const fd = new FormData();
    fd.append("audio", file, file.name || "audio.wav");
    fd.append("adapter_dir", document.getElementById("adapterDir").value);
    fd.append("device", document.getElementById("device").value);
    fd.append("language", document.getElementById("language").value);
    fd.append("beam_size", document.getElementById("beam").value);
    fd.append("temperature", document.getElementById("temp").value);
    return fd;
  }

  async function transcribeFile(file, message) {
    transcribeBtn.disabled = true;
    uploadBtn.disabled = true;
    setStatus(message);

    try {
      const resp = await fetch("/api/transcribe", { method: "POST", body: buildFormData(file) });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || data.detail || resp.statusText);
      }
      setTranscript(data.text || "", data.timestamp);
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
    mediaRecorder.onstop = () => setStatus("Recorded. Ready to transcribe.");

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
    wavFile.value = "";
    wavFile.click();
  };

  wavFile.onchange = async () => {
    const f = wavFile.files && wavFile.files[0];
    if (!f) return;
    copyBtn.disabled = true;
    saveBtn.disabled = true;
    await transcribeFile(f, "Uploading...");
    wavFile.value = "";
  };

  transcribeBtn.onclick = async () => {
    if (!chunks.length) {
      setStatus("Record something first.");
      return;
    }

    setStatus("Transcribing mic recording...");
    transcribeBtn.disabled = true;

    const source = chunks.length ? chunks[0] : null;
    const mimeType = source && source.type ? source.type : "audio/webm";
    const blob = new Blob(chunks, { type: mimeType || "application/octet-stream" });
    const arrayBuffer = await blob.arrayBuffer();

    const filename = source && source.name ? source.name : "audio.webm";
    const file = new File([arrayBuffer], filename, { type: mimeType || "audio/webm" });

    await transcribeFile(file, "Transcribing mic recording...");
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

    const resp = await fetch("/api/download_model", { method: "POST", body: fd });
    const data = await resp.json();

    if (data.error) {
      setStatus("Error: " + data.error);
    } else {
      setStatus(`Downloaded ${data.model_id} to HF cache.`);
    }

    downloadModelBtn.disabled = false;
  };
});
