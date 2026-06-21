let currentJobId = null;

const streamImg = document.getElementById('stream');
const streamPlaceholder = document.getElementById('streamPlaceholder');
const statusEl = document.getElementById('status');
const outputPathEl = document.getElementById('outputPath');
const pillModeEl = document.getElementById('pillMode');
const pillConnEl = document.getElementById('pillConn');

const btnCamera = document.getElementById('btnCamera');
const btnStop = document.getElementById('btnStop');
const btnUpload = document.getElementById('btnUpload');

const videoFile = document.getElementById('videoFile');
const modelSelect = document.getElementById('modelSelect');
const confInput = document.getElementById('confInput');

function setStatus(text) {
  statusEl.textContent = text;
}

function setModePill(modeText) {
  if (pillModeEl) pillModeEl.textContent = `Mode: ${modeText}`;
}

function setConnPill(text, tone = 'muted') {
  if (!pillConnEl) return;
  pillConnEl.textContent = text;
  pillConnEl.classList.toggle('pill--muted', tone === 'muted');
}

function disableControls({ camera = false, stop = false, upload = false } = {}) {
  if (btnCamera) btnCamera.disabled = camera;
  if (btnStop) btnStop.disabled = stop;
  if (btnUpload) btnUpload.disabled = upload;
}

function clearStream() {
  currentJobId = null;
  if (streamImg) streamImg.style.display = 'none';
  if (streamImg) streamImg.removeAttribute('src');
  if (streamPlaceholder) streamPlaceholder.style.display = 'flex';
}

function startStreaming(jobId) {
  currentJobId = jobId;

  if (streamPlaceholder) streamPlaceholder.style.display = 'none';
  if (streamImg) streamImg.style.display = 'block';

  outputPathEl.textContent = '';
  setStatus('Running…');
  setModePill('running');

  // MJPEG stream via server.
  streamImg.src = `/video_feed?job_id=${encodeURIComponent(jobId)}&t=${Date.now()}`;
}

function friendlyError(e) {
  if (!e) return 'Unexpected error';
  if (typeof e === 'string') return e;
  if (e.message) return e.message;
  return String(e);
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) {
    throw new Error(data.error || `Request failed: ${res.status}`);
  }
  return data;
}

btnCamera.addEventListener('click', async () => {
  try {
    clearStream();

    disableControls({ camera: true, stop: true, upload: true });
    setConnPill('Starting camera…', 'muted');
    setStatus('Starting camera…');
    setModePill('starting');

    const model = modelSelect.value;
    const conf = parseFloat(confInput.value || '0.4');

    const data = await postJson('/start_camera', { model, conf });
    if (!data.ok) throw new Error(data.error || 'Camera start failed');

    // Enable stop once started
    disableControls({ camera: true, stop: false, upload: true });

    startStreaming(data.job_id);
    setConnPill('Streaming', 'muted');
  } catch (e) {
    console.error(e);
    setModePill('idle');
    setConnPill('Error', 'muted');
    setStatus('Error: ' + friendlyError(e));

    // Restore controls
    disableControls({ camera: false, stop: true, upload: false });
  }
});

btnStop.addEventListener('click', async () => {
  try {
    setStatus('Stopping…');
    setModePill('stopping');

    await fetch('/stop', { method: 'POST' });

    clearStream();
    setStatus('Stopped');
    setModePill('idle');
    setConnPill('Ready', 'muted');

    disableControls({ camera: false, stop: true, upload: false });
  } catch (e) {
    console.error(e);
    setStatus('Error: ' + friendlyError(e));
    setModePill('idle');
    disableControls({ camera: false, stop: true, upload: false });
  }
});

btnUpload.addEventListener('click', async () => {
  try {
    const file = videoFile.files[0];
    if (!file) {
      setStatus('Choose a video file first');
      return;
    }

    clearStream();

    disableControls({ camera: true, stop: true, upload: true });
    setConnPill('Uploading & starting…', 'muted');
    setStatus('Uploading…');
    setModePill('uploading');

    const model = modelSelect.value;
    const conf = parseFloat(confInput.value || '0.4');

    const fd = new FormData();
    fd.append('video', file);
    fd.append('model', model);
    fd.append('conf', String(conf));

    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      throw new Error(data.error || 'Upload failed');
    }

    disableControls({ camera: true, stop: false, upload: true });

    startStreaming(data.job_id);
    setConnPill('Streaming', 'muted');

    if (data.output) {
      outputPathEl.textContent = 'Tracked output saved at: ' + data.output;
    }
  } catch (e) {
    console.error(e);
    setModePill('idle');
    setConnPill('Error', 'muted');
    setStatus('Error: ' + friendlyError(e));
    disableControls({ camera: false, stop: true, upload: false });
  }
});

// Initialize UI
(function init() {
  disableControls({ camera: false, stop: true, upload: false });
  setModePill('idle');
  setConnPill('Ready');
  clearStream();
  setStatus('Waiting for action…');
})();

