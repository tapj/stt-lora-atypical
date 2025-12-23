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
    console.log("transcribe start", file && file.name);
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
