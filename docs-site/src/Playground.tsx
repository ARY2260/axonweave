import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { FiZap, FiLink, FiActivity, FiAlertTriangle, FiPause, FiPlay, FiRotateCw } from 'react-icons/fi';

/* ------------------------------------------------------------------ */
/*  Demo simulation — NOT real biological data.                       */
/*  Uses LIF (Leaky Integrate-and-Fire) neuron model on a small      */
/*  random graph. This is a pedagogical toy to illustrate how a       */
/*  connectome-based spiking network processes signals.               */
/* ------------------------------------------------------------------ */

interface Neuron {
  id: number;
  x: number;
  y: number;
  v: number;
  vRest: number;
  vThresh: number;
  tauM: number;
  refrac: number;
  region: string;
  lastSpikeTick: number;
}

interface Synapse {
  pre: number;
  post: number;
  weight: number;
  lastActiveTick: number;
}

interface Spike {
  neuron: number;
  time: number;
}

const REGIONS = ['visual', 'motor', 'association', 'sensory'] as const;
const REGION_COLORS: Record<string, string> = {
  visual: '#4d8fd6',
  motor: '#ee4c2c',
  association: '#8dc4d8',
  sensory: '#3ca878',
};
const REGION_LABELS: Record<string, string> = {
  visual: 'Visual',
  motor: 'Motor',
  association: 'Association',
  sensory: 'Sensory',
};

// --- Seeded RNG ---
function mulberry32(a: number) {
  return () => {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Generate network ---
function generateNetwork(n: number, density: number, seed: number) {
  const rng = mulberry32(seed);
  const neurons: Neuron[] = [];
  const angleStep = (2 * Math.PI) / n;
  const cx = 0.5, cy = 0.5, r = 0.38;

  for (let i = 0; i < n; i++) {
    const region = REGIONS[Math.floor(rng() * REGIONS.length)];
    neurons.push({
      id: i,
      x: cx + r * Math.cos(angleStep * i - Math.PI / 2) + (rng() - 0.5) * 0.06,
      y: cy + r * Math.sin(angleStep * i - Math.PI / 2) + (rng() - 0.5) * 0.06,
      v: -0.070,
      vRest: -0.070,
      vThresh: -0.050 + (rng() - 0.5) * 0.004,
      tauM: 0.010 + rng() * 0.010,
      refrac: 0,
      region,
      lastSpikeTick: -9999,
    });
  }

  const synapses: Synapse[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (rng() < density) {
        const isExcit = rng() > 0.2;
        synapses.push({
          pre: i,
          post: j,
          weight: (rng() * 0.008 + 0.001) * (isExcit ? 1 : -0.5),
          lastActiveTick: -9999,
        });
      }
    }
  }
  return { neurons, synapses };
}

// --- LIF simulation step ---
function simulateStep(
  neurons: Neuron[],
  synapses: Synapse[],
  inputCurrent: number[],
  dt: number,
  tick: number,
): Spike[] {
  const newSpikes: Spike[] = [];

  for (const n of neurons) {
    if (n.refrac > 0) {
      n.refrac -= dt;
      continue;
    }
    const I = inputCurrent[n.id] || 0;
    const dv = (-(n.v - n.vRest) / n.tauM + I * 1000) * dt;
    n.v += dv;
    n.v = Math.min(n.v, 0.030);

    if (n.v >= n.vThresh) {
      newSpikes.push({ neuron: n.id, time: tick });
      n.v = n.vRest;
      n.refrac = 0.003;
      n.lastSpikeTick = tick;
    }
  }

  // Propagate spikes through synapses
  for (const sp of newSpikes) {
    const outSynapses = synapses.filter(s => s.pre === sp.neuron);
    for (const syn of outSynapses) {
      neurons[syn.post].v += syn.weight;
      syn.lastActiveTick = tick;
    }
  }

  return newSpikes;
}

// --- Input signal generators ---
function generateInput(t: number, n: number, mode: string, freq: number, amp: number): number[] {
  const input = new Array(n).fill(0);
  const period = 1 / freq;
  const phase = (t % period) / period;
  const nInput = Math.min(4, n);

  switch (mode) {
    case 'sine':
      for (let i = 0; i < nInput; i++) {
        input[i] = amp * Math.sin(2 * Math.PI * freq * t + i * 0.7) * 0.012;
      }
      break;
    case 'pulse':
      if (phase < 0.15) {
        for (let i = 0; i < nInput; i++) {
          input[i] = amp * 0.015;
        }
      }
      break;
    case 'burst':
      if (phase < 0.05 || (phase > 0.2 && phase < 0.25)) {
        for (let i = 0; i < Math.min(3, n); i++) {
          input[i] = amp * 0.018;
        }
      }
      break;
    case 'ramp':
      for (let i = 0; i < nInput; i++) {
        input[i] = amp * phase * 0.015;
      }
      break;
  }
  return input;
}

// --- Get logical canvas dimensions (CSS size, not physical pixels) ---
function getCanvasLogicalSize(canvas: HTMLCanvasElement): { w: number; h: number } {
  const parent = canvas.parentElement;
  if (!parent) return { w: 300, h: 200 };
  const rect = parent.getBoundingClientRect();
  return { w: rect.width, h: rect.height };
}

// --- Resize canvas to match container (DPR-aware) ---
function resizeCanvas(canvas: HTMLCanvasElement) {
  const { w, h } = getCanvasLogicalSize(canvas);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ====================================================================
// DRAWING FUNCTIONS — all use logical (CSS) dimensions
// ====================================================================

function drawNeuronMap(
  canvas: HTMLCanvasElement,
  neurons: Neuron[],
  synapses: Synapse[],
  currentTick: number,
  dark: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { w: W, h: H } = getCanvasLogicalSize(canvas);
  const bg = dark ? '#0d1117' : '#f8f9fa';
  const textColor = dark ? '#8b949e' : '#5f6b76';

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const pad = 24;
  const sx = (v: number) => pad + v * (W - 2 * pad);
  const sy = (v: number) => pad + v * (H - 2 * pad);

  // Draw synapses — glow if recently active
  for (const syn of synapses) {
    const pre = neurons[syn.pre];
    const post = neurons[syn.post];
    const age = currentTick - syn.lastActiveTick;
    const isActive = age >= 0 && age < 60;
    const isExcit = syn.weight > 0;

    ctx.beginPath();
    ctx.moveTo(sx(pre.x), sy(pre.y));
    ctx.lineTo(sx(post.x), sy(post.y));

    if (isActive) {
      const alpha = 0.2 + 0.6 * (1 - age / 60);
      ctx.strokeStyle = isExcit
        ? `rgba(77,143,214,${alpha})`
        : `rgba(238,76,44,${alpha})`;
      ctx.lineWidth = 1.5 + (1 - age / 60) * 1.5;
    } else {
      ctx.strokeStyle = dark ? 'rgba(48,54,61,0.25)' : 'rgba(200,206,212,0.3)';
      ctx.lineWidth = 0.5;
    }
    ctx.stroke();
  }

  // Draw neurons
  for (const n of neurons) {
    const x = sx(n.x), y = sy(n.y);
    const spikeAge = currentTick - n.lastSpikeTick;
    const spiked = spikeAge >= 0 && spikeAge < 80;
    const vNorm = Math.max(0, Math.min(1, (n.v - n.vRest) / (n.vThresh - n.vRest)));
    const baseColor = REGION_COLORS[n.region] || '#8dc4d8';

    // Spike glow (fading)
    if (spiked) {
      const glowAlpha = 0.35 * (1 - spikeAge / 80);
      const glowR = 12 + spikeAge * 0.05;
      ctx.beginPath();
      ctx.arc(x, y, glowR, 0, Math.PI * 2);
      ctx.fillStyle = baseColor;
      ctx.globalAlpha = glowAlpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Neuron body
    const r = spiked ? 7 : 5.5;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = spiked ? baseColor : (dark ? '#30363d' : '#d0d7de');
    ctx.fill();
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = spiked ? 2.5 : 1;
    ctx.stroke();

    // Membrane potential bar
    const barW = 3, barH = 18;
    const barX = x + r + 4, barY = y - barH / 2;
    // Track background
    ctx.fillStyle = dark ? '#161b22' : '#eaecf0';
    ctx.fillRect(barX, barY, barW, barH);
    // Fill based on voltage
    const fillH = Math.max(1, barH * vNorm);
    const fillColor = vNorm > 0.85 ? '#ee4c2c' : vNorm > 0.5 ? '#e8a13c' : baseColor;
    ctx.fillStyle = fillColor;
    ctx.fillRect(barX, barY + barH - fillH, barW, fillH);
    // Border
    ctx.strokeStyle = dark ? '#30363d' : '#d0d7de';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(barX, barY, barW, barH);

    // Neuron ID (every 4th neuron)
    if (n.id % 4 === 0) {
      ctx.fillStyle = textColor;
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`N${n.id}`, x, y - r - 4);
    }
  }

  // Legend
  ctx.font = '11px IBM Plex Sans, sans-serif';
  ctx.textAlign = 'left';
  const legendY = H - 10;
  let legendX = 10;
  for (const region of REGIONS) {
    ctx.fillStyle = REGION_COLORS[region];
    ctx.beginPath();
    ctx.arc(legendX + 5, legendY - 3, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = textColor;
    const label = REGION_LABELS[region];
    ctx.fillText(label, legendX + 14, legendY);
    legendX += ctx.measureText(label).width + 24;
  }
}

function drawRaster(
  canvas: HTMLCanvasElement,
  spikes: Spike[],
  currentTick: number,
  nNeurons: number,
  dark: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { w: W, h: H } = getCanvasLogicalSize(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#0d1117' : '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = dark ? 'rgba(48,54,61,0.35)' : 'rgba(200,206,212,0.4)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = (H / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Time axis labels
  const textColor = dark ? '#8b949e' : '#5f6b76';
  ctx.fillStyle = textColor;
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  const windowMs = 600;
  for (let i = 0; i <= 4; i++) {
    const ms = Math.round((currentTick - windowMs) + (windowMs / 4) * i);
    ctx.fillText(`${(ms / 1000).toFixed(1)}s`, (W / 4) * i, H - 3);
  }

  // Draw spikes
  const startTick = Math.max(0, currentTick - windowMs);
  for (const s of spikes) {
    if (s.time < startTick) continue;
    const x = ((s.time - startTick) / windowMs) * W;
    const y = (s.neuron / Math.max(1, nNeurons - 1)) * (H - 16) + 4;
    const region = REGIONS[s.neuron % REGIONS.length];
    const age = (currentTick - s.time) / windowMs;
    ctx.fillStyle = REGION_COLORS[region];
    ctx.globalAlpha = Math.max(0.25, 1 - age);
    ctx.fillRect(x, y - 1.5, 2.5, 3);
    ctx.globalAlpha = 1;
  }

  // Y-axis labels
  ctx.fillStyle = textColor;
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`N${nNeurons - 1}`, 2, 14);
  ctx.fillText('N0', 2, H - 16);
}

function drawVoltageTrace(
  canvas: HTMLCanvasElement,
  traces: number[][],
  dark: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { w: W, h: H } = getCanvasLogicalSize(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#0d1117' : '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = dark ? 'rgba(48,54,61,0.35)' : 'rgba(200,206,212,0.4)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = (H / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  const vMin = -0.075, vMax = -0.035;
  const traceLen = 250;
  const colors = ['#4d8fd6', '#ee4c2c', '#8dc4d8', '#3ca878', '#c792ea'];

  for (let t = 0; t < Math.min(5, traces.length); t++) {
    const trace = traces[t];
    if (!trace || trace.length === 0) continue;
    ctx.beginPath();
    ctx.strokeStyle = colors[t];
    ctx.lineWidth = 1.3;
    ctx.globalAlpha = 0.9;
    const len = Math.min(traceLen, trace.length);
    const start = trace.length - len;
    for (let i = 0; i < len; i++) {
      const x = (i / (traceLen - 1)) * W;
      const v = trace[start + i];
      const y = H - 8 - ((v - vMin) / (vMax - vMin)) * (H - 16);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Labels
  const textColor = dark ? '#8b949e' : '#5f6b76';
  ctx.fillStyle = textColor;
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('-35mV', W - 4, 14);
  ctx.fillText('-75mV', W - 4, H - 6);

  // Threshold line
  const threshY = H - 8 - ((-0.050 - vMin) / (vMax - vMin)) * (H - 16);
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = '#ee4c2c';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, threshY); ctx.lineTo(W, threshY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ee4c2c';
  ctx.textAlign = 'left';
  ctx.fillText('threshold', 4, threshY - 4);
}

function drawInputWaveform(
  canvas: HTMLCanvasElement,
  mode: string,
  freq: number,
  amp: number,
  currentT: number,
  dark: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { w: W, h: H } = getCanvasLogicalSize(canvas);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#0d1117' : '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = dark ? 'rgba(48,54,61,0.3)' : 'rgba(200,206,212,0.35)';
  ctx.lineWidth = 0.5;
  const midY = H / 2;
  ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

  // Draw waveform
  const windowSec = 1.0 / freq * 3; // show 3 periods
  const tStart = Math.max(0, currentT - windowSec);
  ctx.beginPath();
  ctx.strokeStyle = '#8dc4d8';
  ctx.lineWidth = 1.5;
  const nPts = 200;
  for (let i = 0; i < nPts; i++) {
    const t = tStart + (i / nPts) * windowSec;
    const input = generateInput(t, 1, mode, freq, amp);
    const val = input[0] || 0;
    const x = (i / nPts) * W;
    // Map val to canvas: 0 = midY, positive = up
    const maxVal = 0.04;
    const y = midY - (val / maxVal) * (H / 2 - 8);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Labels
  const textColor = dark ? '#8b949e' : '#5f6b76';
  ctx.fillStyle = textColor;
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(mode.toUpperCase(), 4, 14);
  ctx.fillText(`${freq}Hz`, 4, 26);
}

// --- Equivalence code ---
function generateEquivalenceCode(
  mode: string, freq: number, amp: number, nNeurons: number, tauM: number,
  density: number, seed: number,
): string {
  return `# AxonWeave equivalent (NOT running here — demo only)
import axonweave
from axonweave.dynamics import LIF
from axonweave.runtime import ConnectomeRuntime

brain = axonweave.load("male-cns:v1.0")

# LIF dynamics — same parameters as the demo
dynamics = LIF(
    tau_membrane=${tauM.toFixed(3)},   # membrane time constant (s)
    v_threshold=-0.050,  # spike threshold (V)
    v_rest=-0.070,       # resting potential (V)
    dt=0.001,            # 1 ms timestep
)

# Runtime on the real substrate
runtime = ConnectomeRuntime(
    brain.graph,
    dynamics=dynamics,
    n_neurons=${nNeurons},   # demo topology
)

runtime.reset_state()

# Input: ${mode} wave, ${freq} Hz, amplitude ${amp.toFixed(2)}
# In the demo: neurons 0-${Math.min(3, nNeurons - 1)} receive input
# In real use: input maps to sensory neuron regions
for t in range(1000):
    x_t = generate_input(t * 0.001)  # shape: [batch, features]
    y_t = runtime.step(x_t)          # 1 ms step

# Access spike state
state = runtime.get_state()
spikes = state.neuron.spiked
print(f"Active neurons: {spikes.sum()}/{${nNeurons}}")

# Topology: ${nNeurons} neurons, density ${density.toFixed(2)}, seed ${seed}
# Real MaleCNS: 166,700 neurons, millions of synapses`;
}

// ====================================================================
// MAIN COMPONENT
// ====================================================================

interface PlaygroundProps { dark: boolean; }

export default function Playground({ dark }: PlaygroundProps) {
  const [nNeurons, setNNeurons] = useState(24);
  const [density, setDensity] = useState(0.15);
  const [seed, setSeed] = useState(42);
  const [mode, setMode] = useState('sine');
  const [freq, setFreq] = useState(3);
  const [amplitude, setAmplitude] = useState(1.0);
  const [tauM, setTauM] = useState(0.015);
  const [speed, setSpeed] = useState(1.0);
  const [running, setRunning] = useState(false);

  // Stats — use state so they trigger re-renders
  const [tick, setTick] = useState(0);
  const [spikeCount, setSpikeCount] = useState(0);
  const [firingRate, setFiringRate] = useState(0);
  const [activeNeurons, setActiveNeurons] = useState(0);

  // Canvas refs
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const rasterCanvasRef = useRef<HTMLCanvasElement>(null);
  const voltageCanvasRef = useRef<HTMLCanvasElement>(null);
  const inputCanvasRef = useRef<HTMLCanvasElement>(null);

  // Mutable simulation state (refs for performance)
  const netRef = useRef(generateNetwork(nNeurons, density, seed));
  const spikesRef = useRef<Spike[]>([]);
  const tickRef = useRef(0);
  const tracesRef = useRef<number[][]>([]);
  const animRef = useRef<number>(0);
  const lastFrameRef = useRef(0);
  const rateWindowRef = useRef<number[]>([]);
  const totalSpikeRef = useRef(0);

  // Rebuild network
  const rebuild = useCallback(() => {
    const net = generateNetwork(nNeurons, density, seed);
    netRef.current = net;
    spikesRef.current = [];
    tickRef.current = 0;
    tracesRef.current = net.neurons.map(n => [n.v]);
    rateWindowRef.current = [];
    totalSpikeRef.current = 0;
    setTick(0);
    setSpikeCount(0);
    setFiringRate(0);
    setActiveNeurons(0);
  }, [nNeurons, density, seed]);

  useEffect(rebuild, [rebuild]);

  // Resize canvases
  useEffect(() => {
    const canvases = [mapCanvasRef.current, rasterCanvasRef.current, voltageCanvasRef.current, inputCanvasRef.current];
    const resize = () => canvases.forEach(c => { if (c) resizeCanvas(c); });
    resize();
    // Initial draw
    if (mapCanvasRef.current) {
      resizeCanvas(mapCanvasRef.current);
      drawNeuronMap(mapCanvasRef.current, netRef.current.neurons, netRef.current.synapses, 0, dark);
    }
    if (rasterCanvasRef.current) {
      resizeCanvas(rasterCanvasRef.current);
      drawRaster(rasterCanvasRef.current, [], 0, nNeurons, dark);
    }
    if (voltageCanvasRef.current) {
      resizeCanvas(voltageCanvasRef.current);
      drawVoltageTrace(voltageCanvasRef.current, netRef.current.neurons.slice(0, 5).map(n => [n.v]), dark);
    }
    if (inputCanvasRef.current) {
      resizeCanvas(inputCanvasRef.current);
      drawInputWaveform(inputCanvasRef.current, mode, freq, amplitude, 0, dark);
    }

    const ro = new ResizeObserver(resize);
    canvases.forEach(c => { if (c?.parentElement) ro.observe(c.parentElement); });
    return () => ro.disconnect();
  }, [dark, nNeurons, mode, freq, amplitude]);

  // Simulation + render loop
  useEffect(() => {
    if (!running) {
      cancelAnimationFrame(animRef.current);
      return;
    }

    const loop = (ts: number) => {
      if (ts - lastFrameRef.current > 16) {
        const stepsPerFrame = Math.max(1, Math.round(speed * 4));
        let newSpikesAll: Spike[] = [];

        for (let i = 0; i < stepsPerFrame; i++) {
          const t = tickRef.current;
          const net = netRef.current;
          const input = generateInput(t / 1000, net.neurons.length, mode, freq, amplitude);
          const newSpikes = simulateStep(net.neurons, net.synapses, input, 0.001, t);
          newSpikesAll.push(...newSpikes);
          spikesRef.current.push(...newSpikes);
          tickRef.current += 1;

          // Record traces
          for (let j = 0; j < Math.min(5, net.neurons.length); j++) {
            tracesRef.current[j] = tracesRef.current[j] || [];
            tracesRef.current[j].push(net.neurons[j].v);
            if (tracesRef.current[j].length > 400) tracesRef.current[j].shift();
          }
        }

        totalSpikeRef.current += newSpikesAll.length;

        // Firing rate (spikes/sec over last 200ms window)
        rateWindowRef.current.push(newSpikesAll.length);
        if (rateWindowRef.current.length > 200) rateWindowRef.current.shift();
        const totalInWindow = rateWindowRef.current.reduce((a, b) => a + b, 0);

        // Active neurons (spiked in last 200ms)
        const activeSet = new Set<number>();
        const ct = tickRef.current;
        for (const s of spikesRef.current) {
          if (ct - s.time < 200) activeSet.add(s.neuron);
        }

        // Update React state (batched)
        setTick(tickRef.current);
        setSpikeCount(totalSpikeRef.current);
        setFiringRate(Math.round(totalInWindow * 5));
        setActiveNeurons(activeSet.size);

        // Draw canvases
        const net = netRef.current;
        if (mapCanvasRef.current) {
          drawNeuronMap(mapCanvasRef.current, net.neurons, net.synapses, tickRef.current, dark);
        }
        if (rasterCanvasRef.current) {
          drawRaster(rasterCanvasRef.current, spikesRef.current, tickRef.current, net.neurons.length, dark);
        }
        if (voltageCanvasRef.current) {
          drawVoltageTrace(voltageCanvasRef.current, tracesRef.current, dark);
        }
        if (inputCanvasRef.current) {
          drawInputWaveform(inputCanvasRef.current, mode, freq, amplitude, tickRef.current / 1000, dark);
        }

        // Prune old spikes (keep last 2000 for raster)
        if (spikesRef.current.length > 2000) {
          spikesRef.current.splice(0, spikesRef.current.length - 2000);
        }

        lastFrameRef.current = ts;
      }
      animRef.current = requestAnimationFrame(loop);
    };

    lastFrameRef.current = 0;
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, dark, mode, freq, amplitude, speed]);

  const code = useMemo(
    () => generateEquivalenceCode(mode, freq, amplitude, nNeurons, tauM, density, seed),
    [mode, freq, amplitude, nNeurons, tauM, density, seed],
  );

  const canvasStyle = useMemo(() => ({
    width: '100%', height: '100%', display: 'block' as const,
  }), []);

  // Presets: pre-built demo scenarios
  const presets: Record<string, { n: number; d: number; s: number; m: string; f: number; a: number; t: number; label: string; desc: string }> = {
    default: { n: 24, d: 0.15, s: 42, m: 'sine', f: 3, a: 1.0, t: 15, label: 'Default', desc: 'Balanced network with sine input' },
    sparse: { n: 16, d: 0.08, s: 7, m: 'pulse', f: 2, a: 1.5, t: 20, label: 'Sparse Pulse', desc: 'Low density, pulse-driven, slow dynamics' },
    dense_burst: { n: 32, d: 0.25, s: 99, m: 'burst', f: 5, a: 2.0, t: 10, label: 'Dense Burst', desc: 'High connectivity, fast bursting input' },
    minimal: { n: 8, d: 0.2, s: 1, m: 'sine', f: 1, a: 0.8, t: 15, label: 'Minimal', desc: 'Tiny network, easy to observe individual neurons' },
    cortical: { n: 36, d: 0.18, s: 55, m: 'ramp', f: 4, a: 1.2, t: 12, label: 'Cortical-like', desc: 'Larger network with ramping input' },
  };
  const [preset, setPreset] = useState('default');

  const applyPreset = (key: string) => {
    const p = presets[key];
    if (!p) return;
    setPreset(key);
    setRunning(false);
    setNNeurons(p.n);
    setDensity(p.d);
    setSeed(p.s);
    setMode(p.m);
    setFreq(p.f);
    setAmplitude(p.a);
    setTauM(p.t / 1000);
  };

  return (
    <div className="playground-wrapper">
      {/* Banner */}
      <div className="playground-banner">
        <span className="playground-badge">DEMO</span>
        <span>This is a simplified simulation for exploration. It uses random weights and a toy graph — not the real MaleCNS connectome.</span>
      </div>

      {/* Presets */}
      <div className="playground-presets">
        <span className="presets-label">Quick demos</span>
        <div className="presets-row">
          {Object.entries(presets).map(([key, p]) => (
            <button key={key} className={`preset-btn ${preset === key ? 'active' : ''}`}
              onClick={() => applyPreset(key)} title={p.desc}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="playground-controls">
        <div className="control-group">
          <label className="control-label">Network</label>
          <div className="control-row">
            <span className="control-name">Neurons</span>
            <input type="range" min="8" max="40" step="1" value={nNeurons}
              onChange={e => setNNeurons(Number(e.target.value))} />
            <span className="control-value">{nNeurons}</span>
          </div>
          <div className="control-row">
            <span className="control-name">Density</span>
            <input type="range" min="0.05" max="0.35" step="0.01" value={density}
              onChange={e => setDensity(Number(e.target.value))} />
            <span className="control-value">{density.toFixed(2)}</span>
          </div>
          <div className="control-row">
            <span className="control-name">Seed</span>
            <input type="range" min="1" max="100" step="1" value={seed}
              onChange={e => setSeed(Number(e.target.value))} />
            <span className="control-value">{seed}</span>
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Dynamics</label>
          <div className="control-row">
            <span className="control-name">τ_m (ms)</span>
            <input type="range" min="5" max="30" step="1" value={tauM * 1000}
              onChange={e => setTauM(Number(e.target.value) / 1000)} />
            <span className="control-value">{(tauM * 1000).toFixed(0)}</span>
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Input Signal</label>
          <div className="control-row">
            <span className="control-name">Mode</span>
            <select value={mode} onChange={e => setMode(e.target.value)}>
              <option value="sine">Sine wave</option>
              <option value="pulse">Pulse train</option>
              <option value="burst">Burst</option>
              <option value="ramp">Ramp</option>
            </select>
          </div>
          <div className="control-row">
            <span className="control-name">Freq (Hz)</span>
            <input type="range" min="0.5" max="10" step="0.5" value={freq}
              onChange={e => setFreq(Number(e.target.value))} />
            <span className="control-value">{freq}</span>
          </div>
          <div className="control-row">
            <span className="control-name">Amplitude</span>
            <input type="range" min="0.1" max="3" step="0.1" value={amplitude}
              onChange={e => setAmplitude(Number(e.target.value))} />
            <span className="control-value">{amplitude.toFixed(1)}</span>
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Playback</label>
          <div className="control-row">
            <span className="control-name">Speed</span>
            <input type="range" min="0.2" max="4" step="0.2" value={speed}
              onChange={e => setSpeed(Number(e.target.value))} />
            <span className="control-value">{speed.toFixed(1)}x</span>
          </div>
          <div className="control-row control-buttons">
            <button className={`play-btn ${running ? 'running' : ''}`}
              onClick={() => setRunning(!running)}>{running ? <><FiPause size={14}/> Pause</> : <><FiPlay size={14}/> Run</>}</button>
            <button className="reset-btn" onClick={() => { setRunning(false); rebuild(); }}><><FiRotateCw size={14}/> Reset</></button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="playground-stats">
        <div className="stat">
          <span className="stat-label">Time</span>
          <span className="stat-value">{(tick / 1000).toFixed(2)}s</span>
        </div>
        <div className="stat">
          <span className="stat-label">Spikes</span>
          <span className="stat-value">{spikeCount}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Firing Rate</span>
          <span className="stat-value">{firingRate} Hz</span>
        </div>
        <div className="stat">
          <span className="stat-label">Active</span>
          <span className="stat-value">{activeNeurons}/{nNeurons}</span>
        </div>
      </div>

      {/* Visualization */}
      <div className="playground-viz">
        <div className="viz-panel neuron-map">
          <div className="viz-title">Neuron Map — neurons glow when spiking, potential bars show membrane voltage, synapses flash on activation</div>
          <div className="canvas-container">
            <canvas ref={mapCanvasRef} style={canvasStyle} />
          </div>
        </div>
        <div className="viz-side">
          <div className="viz-panel raster-panel">
            <div className="viz-title">Spike Raster (last 600ms)</div>
            <div className="canvas-container">
              <canvas ref={rasterCanvasRef} style={canvasStyle} />
            </div>
          </div>
          <div className="viz-panel voltage-panel">
            <div className="viz-title">Membrane Traces (5 neurons)</div>
            <div className="canvas-container">
              <canvas ref={voltageCanvasRef} style={canvasStyle} />
            </div>
          </div>
          <div className="viz-panel input-panel">
            <div className="viz-title">Input Waveform to N0-N3</div>
            <div className="canvas-container">
              <canvas ref={inputCanvasRef} style={canvasStyle} />
            </div>
          </div>
        </div>
      </div>

      {/* Code panel */}
      <div className="playground-code">
        <div className="code-panel-header">
          <span className="code-panel-title">Equivalent AxonWeave Code</span>
          <span className="code-panel-note">This code shows the real API — the simulation above is a browser-side approximation.</span>
        </div>
        <pre><code>{code}</code></pre>
      </div>

      {/* How it works */}
      <div className="playground-explain">
        <h3>How this simulation works</h3>
        <div className="explain-grid">
          <div className="explain-card">
            <div className="explain-icon"><FiZap size={24}/></div>
            <h4>LIF Neuron Model</h4>
            <p>Each neuron integrates input current into a membrane potential <code>V</code>. When <code>V</code> exceeds the threshold, it fires a spike and resets. The time constant <code>τ_m</code> controls how fast the membrane leaks back toward rest.</p>
          </div>
          <div className="explain-card">
            <div className="explain-icon"><FiLink size={24}/></div>
            <h4>Synaptic Propagation</h4>
            <p>When a neuron spikes, its signal propagates through weighted synapses (excitatory <span style={{color:'#4d8fd6'}}>blue</span> / inhibitory <span style={{color:'#ee4c2c'}}>red</span>). Connected neurons receive the weight as a voltage bump on the next timestep.</p>
          </div>
          <div className="explain-card">
            <div className="explain-icon"><FiActivity size={24}/></div>
            <h4>Neuron Regions</h4>
            <p>Neurons are assigned to brain regions (Visual, Motor, Association, Sensory). In the real AxonWeave, these come from MaleCNS cell-type annotations — not random assignment.</p>
          </div>
          <div className="explain-card">
            <div className="explain-icon"><FiAlertTriangle size={24}/></div>
            <h4>This is a Demo</h4>
            <p>The real MaleCNS connectome has <strong>166,700 neurons</strong> and millions of synapses. This browser demo uses {nNeurons} neurons with random topology. For real substrate access, install AxonWeave.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
