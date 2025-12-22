let mediaRecorder = null;
let chunks = [];
let lastTranscript = "";

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
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

transcribeBtn.onclick = async () => {
  setStatus("Transcribing...");
  transcribeBtn.disabled = true;

  const blob = new Blob(chunks, { type: "audio/webm" });
  const arrayBuffer = await blob.arrayBuffer();

  // Backend expects wav bytes. Browser gives webm/opus. We send as-is and rely on torchaudio decode.
  // If torchaudio backend cannot decode webm on your system, switch MediaRecorder mimeType to audio/wav via a polyfill,
  // or record to wav in a desktop app. Troubleshooting in README.
  const file = new File([arrayBuffer], "audio.webm", { type: "audio/webm" });

  const fd = new FormData();
  fd.append("audio", file);
  fd.append("adapter_dir", document.getElementById("adapterDir").value);
  fd.append("device", document.getElementById("device").value);
  fd.append("language", document.getElementById("language").value);
  fd.append("beam_size", document.getElementById("beam").value);
  fd.append("temperature", document.getElementById("temp").value);

  const resp = await fetch("/api/transcribe", { method: "POST", body: fd });
  const data = await resp.json();

  if (data.error) {
    setStatus("Error: " + data.error);
    transcribeBtn.disabled = false;
    return;
  }

  lastTranscript = `[${data.timestamp}] ${data.text}`;
  outputEl.value = lastTranscript;
  setStatus("Done.");

  copyBtn.disabled = false;
  saveBtn.disabled = false;
  transcribeBtn.disabled = false;
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
