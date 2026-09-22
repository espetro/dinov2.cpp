// dinov2.cpp wasm demo: load a GGUF, encode two images, show cosine
// similarity and a patch-grid heatmap of patch-to-CLS cosine per image.
// No dependencies; the emscripten module is a plain ES6 import.

import Dinov2Module from './dinov2-wasm.js';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const say = (msg) => { statusEl.textContent += msg + '\n'; };

let Module = null;
let modelHandle = 0;
let modelInfo = null;

// --- model ---------------------------------------------------------------

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

$('loadBtn').addEventListener('click', async () => {
  const url = $('modelUrl').value.trim() || 'model.gguf';
  const prog = $('modelProgress');
  try {
    $('loadBtn').disabled = true;
    say(`loading wasm module (dinov2 ${'(…)'})...`);
    Module = Module || await Dinov2Module();
    statusEl.textContent = statusEl.textContent.replace('(…)', Module.version());

    say(`fetching ${url} ...`);
    prog.hidden = false;
    const t0 = performance.now();
    const bytes = await fetchWithProgress(url, (got, total) => {
      prog.max = total || got;
      prog.value = got;
    });
    say(`fetched ${(bytes.length / 1e6).toFixed(1)} MB in ${(performance.now() - t0).toFixed(0)} ms`);

    say('building model (first call compiles the wasm graph)...');
    const t1 = performance.now();
    const handle = Module.loadModel(bytes);
    if (!handle) throw new Error('dino_model_load_from_buffer failed (see console/stderr)');
    if (modelHandle) Module.freeModel(modelHandle);
    modelHandle = handle;
    modelInfo = Module.modelInfo(handle);
    say(`model ready in ${(performance.now() - t1).toFixed(0)} ms: ` +
        `hidden=${modelInfo.hiddenSize} patch=${modelInfo.patchSize} ` +
        `registers=${modelInfo.nRegisterTokens} classifier=${modelInfo.hasClassifier}`);
    $('compareBtn').disabled = false;
  } catch (err) {
    say(`error: ${err.message}`);
  } finally {
    prog.hidden = true;
    $('loadBtn').disabled = false;
  }
});

// --- image decode ---------------------------------------------------------

const MAX_SIDE = 1200;

// file -> {pixels: Uint8ClampedArray (RGBA), width, height}; also previews it
async function decodeImage(file, previewCanvas) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const cv = previewCanvas;
  cv.width = w; cv.height = h; cv.hidden = false;
  const g = cv.getContext('2d', { willReadFrequently: true });
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const data = g.getImageData(0, 0, w, h);
  return { pixels: data.data, width: w, height: h };
}

// --- math helpers (JS side, per the wasm shim contract) -------------------

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// Per-patch cosine to the CLS token, drawn as a blue->red heatmap.
function drawPatchHeatmap(enc, canvas) {
  if (!enc.patches || !enc.cls || !enc.gridW || !enc.gridH) { canvas.hidden = true; return; }
  const { gridW: gw, gridH: gh, patches, cls, hiddenSize: d } = enc;
  const img = new ImageData(gw, gh);
  for (let p = 0; p < gw * gh; p++) {
    let dot = 0, np = 0, nc = 0;
    for (let i = 0; i < d; i++) {
      const v = patches[p * d + i], c = cls[i];
      dot += v * c; np += v * v; nc += c * c;
    }
    const sim = dot / (Math.sqrt(np) * Math.sqrt(nc) || 1); // roughly [-1, 1]
    const t = Math.max(0, Math.min(1, (sim + 0.2) / 0.8));   // squash for contrast
    img.data[p * 4 + 0] = Math.round(255 * t);
    img.data[p * 4 + 1] = Math.round(64 * (1 - Math.abs(2 * t - 1)));
    img.data[p * 4 + 2] = Math.round(255 * (1 - t));
    img.data[p * 4 + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = gw; tmp.height = gh;
  tmp.getContext('2d').putImageData(img, 0, 0);
  canvas.width = canvas.parentElement.clientWidth || 240;
  canvas.height = Math.round(canvas.width * gh / gw);
  canvas.hidden = false;
  const g = canvas.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

// --- compare --------------------------------------------------------------

const encodings = { A: null, B: null };

async function encodeInto(which) {
  const file = $(`img${which}`).files[0];
  if (!file) { encodings[which] = null; return null; }
  const img = await decodeImage(file, $(`canvas${which}`));
  const t0 = performance.now();
  const enc = Module.encode(modelHandle, img.pixels, img.width, img.height, 4);
  const ms = performance.now() - t0;
  if (!enc.ok) { say(`encode ${which} failed: dino_status=${enc.status}`); return null; }
  enc._ms = ms;
  encodings[which] = enc;
  say(`image ${which}: ${img.width}x${img.height} -> ${enc.nPatches} patches ` +
      `(${enc.gridW}x${enc.gridH}) x${enc.hiddenSize} in ${ms.toFixed(0)} ms`);
  drawPatchHeatmap(enc, $(`heat${which}`));
  return enc;
}

$('compareBtn').addEventListener('click', async () => {
  if (!modelHandle) return;
  $('compareBtn').disabled = true;
  $('result').textContent = '';
  try {
    const a = await encodeInto('A');
    const b = await encodeInto('B');
    if (a && b && a.cls && b.cls) {
      const sim = cosine(a.cls, b.cls);
      $('result').textContent = `cosine(cls_A, cls_B) = ${sim.toFixed(4)}`;
    } else if (a || b) {
      $('result').textContent = 'pick both images for a similarity score';
    } else {
      $('result').textContent = 'pick at least one image';
    }
  } finally {
    $('compareBtn').disabled = false;
  }
});
