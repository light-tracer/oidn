// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Driver for the OIDN WebGPU wasm module. Loads an image (or a synthetic noisy
// sample), runs the RT filter on the GPU, and shows before/after.

let Module = null;         // the Emscripten module instance
let inited = false;        // whether the WebGPU device is created
let curImage = null;       // { width, height, rgba: Uint8ClampedArray } of the noisy input
let curWeightsUrl = null;  // weights blob currently loaded into the module

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const setStatus = (s) => { statusEl.textContent = s; };

// --- sRGB <-> linear (the RT filter's default LDR path works in linear space) ---
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(c) {
  c = Math.min(Math.max(c, 0), 1);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

// --- draw an RGBA byte image onto a canvas ---
function drawRGBA(canvas, width, height, rgba) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
}

// --- synthetic noisy scene: soft gradient + a few disks, plus heavy noise ---
function makeSample(width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const disks = [
    { x: 0.32, y: 0.45, r: 0.20, c: [0.85, 0.25, 0.20] },
    { x: 0.62, y: 0.55, r: 0.16, c: [0.25, 0.75, 0.35] },
    { x: 0.50, y: 0.28, r: 0.11, c: [0.95, 0.85, 0.55] },
  ];
  // deterministic-ish PRNG
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) * 0.5; // ~N(0, ~0.29)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = x / width, v = y / height;
      let r = 0.35 + 0.35 * (1 - v), g = 0.45 + 0.35 * (1 - v), b = 0.65 + 0.30 * (1 - v);
      for (const d of disks) {
        const dx = u - d.x, dy = v - d.y;
        if (dx * dx + dy * dy < d.r * d.r) {
          const nz = Math.sqrt(Math.max(0, d.r * d.r - dx * dx - dy * dy)) / d.r;
          const shade = 0.35 + 0.65 * nz;
          r = d.c[0] * shade; g = d.c[1] * shade; b = d.c[2] * shade;
        }
      }
      // add noise in linear space, then encode to sRGB bytes
      const n = 0.28;
      r = linearToSrgb(Math.max(0, r + gauss() * n));
      g = linearToSrgb(Math.max(0, g + gauss() * n));
      b = linearToSrgb(Math.max(0, b + gauss() * n));
      const i = (y * width + x) * 4;
      rgba[i] = r * 255; rgba[i + 1] = g * 255; rgba[i + 2] = b * 255; rgba[i + 3] = 255;
    }
  }
  return { width, height, rgba };
}

function setInput(img) {
  curImage = img;
  drawRGBA($('src'), img.width, img.height, img.rgba);
  // The output canvas is a WebGPU surface (see renderOutput); resizing clears it
  const dst = $('dst');
  dst.width = img.width; dst.height = img.height;
  $('denoise').disabled = !(Module && curImage);
}

// --- display the denoised result straight from the GPU (no readback) ---
// The wasm module publishes its device and output buffer as Module.oidnOutput;
// a fullscreen pass reads the float3 buffer and writes sRGB to the canvas.
const gpuView = { device: null, ctx: null, pipeline: null, paramsBuf: null };

const kViewWGSL = `
struct P { w: u32, h: u32 };
@group(0) @binding(0) var<storage, read> img : array<f32>;
@group(0) @binding(1) var<uniform> p : P;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}

fn srgb(c0: f32) -> f32 {
  let c = clamp(c0, 0.0, 1.0);
  if (c <= 0.0031308) { return c * 12.92; }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let x = u32(fragPos.x);
  let y = u32(fragPos.y);
  let i = (y * p.w + x) * 3u;
  return vec4f(srgb(img[i]), srgb(img[i + 1u]), srgb(img[i + 2u]), 1.0);
}`;

function renderOutput() {
  const out = Module.oidnOutput;
  if (!out) { console.error('no published output'); return; }
  const device = out.device;
  const canvas = $('dst');

  if (gpuView.device !== device) {
    gpuView.device = device;
    gpuView.ctx = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    gpuView.ctx.configure({ device, format, alphaMode: 'opaque' });
    const module = device.createShaderModule({ code: kViewWGSL });
    gpuView.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
    gpuView.paramsBuf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  if (canvas.width !== out.width || canvas.height !== out.height) {
    canvas.width = out.width;
    canvas.height = out.height;
  }

  device.queue.writeBuffer(gpuView.paramsBuf, 0, new Uint32Array([out.width, out.height, 0, 0]));

  const bindGroup = device.createBindGroup({
    layout: gpuView.pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: out.buffer, offset: out.offset, size: out.byteSize } },
      { binding: 1, resource: { buffer: gpuView.paramsBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: gpuView.ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  pass.setPipeline(gpuView.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
}

async function loadFile(file) {
  const bmp = await createImageBitmap(file);
  // cap size so the demo stays snappy
  const maxDim = 1024;
  const scale = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  setInput({ width: w, height: h, rgba: new Uint8ClampedArray(rgba) });
}

// Fetch a .tza weights blob and hand it to the wasm module (cached per URL)
async function ensureWeights(url) {
  if (curWeightsUrl === url) return true;
  setStatus('loading weights ' + url + '…');
  const resp = await fetch(url);
  if (!resp.ok) { setStatus('failed to fetch ' + url); return false; }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const ptr = Module._malloc(bytes.length);
  Module.HEAPU8.set(bytes, ptr);
  const ok = Module.ccall('oidnWebSetWeights', 'number', ['number', 'number'], [ptr, bytes.length]);
  Module._free(ptr);
  if (!ok) { setStatus('failed to set weights'); return false; }
  curWeightsUrl = url;
  return true;
}

async function denoise() {
  if (!Module || !curImage) return;
  const btn = $('denoise');
  btn.disabled = true;

  try {
    if (!inited) {
      setStatus('initializing WebGPU device…');
      const ok = await Module.ccall('oidnWebInit', 'number', [], [], { async: true });
      if (!ok) { setStatus('failed to create WebGPU device (see console)'); btn.disabled = false; return; }
      inited = true;
    }

    const wantHdr = $('hdr').checked;
    if (!await ensureWeights(wantHdr ? 'weights/rt_hdr.tza' : 'weights/rt_ldr.tza')) {
      btn.disabled = false;
      return;
    }

    const { width, height, rgba } = curImage;
    const hdr = wantHdr ? 1 : 0;
    const n = width * height;
    const floats = new Float32Array(n * 3);
    // sRGB bytes -> linear float RGB
    for (let i = 0; i < n; i++) {
      floats[i * 3 + 0] = srgbToLinear(rgba[i * 4 + 0] / 255);
      floats[i * 3 + 1] = srgbToLinear(rgba[i * 4 + 1] / 255);
      floats[i * 3 + 2] = srgbToLinear(rgba[i * 4 + 2] / 255);
    }

    const bytes = floats.length * 4;
    const inPtr = Module._malloc(bytes);
    Module.HEAPF32.set(floats, inPtr >> 2);

    setStatus('denoising…');
    const t0 = performance.now();
    // Output pointer 0: the result stays on the GPU and gets published as
    // Module.oidnOutput for direct display (no readback)
    const ok = await Module.ccall(
      'oidnWebDenoise',
      'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
      [width, height, inPtr, 0, 0, 0, hdr],
      { async: true });

    if (!ok) { setStatus('denoise failed (see console)'); }
    else {
      renderOutput(); // queue-ordered after the denoise; presents at next frame
      const ms = performance.now() - t0;
      setStatus(`denoised ${width}×${height}, submitted in ${ms.toFixed(1)} ms (display is GPU-direct)`);
      // The completion signal often arrives well after the frame is already
      // presented (Chrome scheduling), so log it instead of showing it
      Module.oidnOutput.device.queue.onSubmittedWorkDone().then(() => {
        console.log(`oidn: gpu completion signal after ${(performance.now() - t0).toFixed(1)} ms total`);
      });
    }

    Module._free(inPtr);
  } catch (e) {
    console.error(e);
    setStatus('error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- wire up UI ---
$('sample').addEventListener('click', () => setInput(makeSample(512, 512)));
$('file').addEventListener('change', (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });
$('denoise').addEventListener('click', denoise);

// --- load the wasm module ---
// Debug helpers: add ?timeops=1 and/or ?oidnverbose=2 to the URL to enable
// OIDN's per-op timing / verbose logging inside the wasm module
const urlParams = new URLSearchParams(location.search);
const moduleArg = {
  preRun: [(m) => {
    if (urlParams.get('timeops')) m.ENV.OIDN_TIME_OPS = '1';
    if (urlParams.get('oidnverbose')) m.ENV.OIDN_VERBOSE = urlParams.get('oidnverbose');
  }],
};

createOIDN(moduleArg).then((m) => {
  Module = m;
  setStatus('ready — generate a sample or load an image');
  setInput(makeSample(512, 512));
  $('denoise').disabled = false;
}).catch((e) => {
  console.error(e);
  setStatus('failed to load wasm module: ' + e.message);
});
