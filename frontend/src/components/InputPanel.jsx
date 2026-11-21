import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Play, Zap, Monitor, Layers, Upload, AlarmClock, FileUp, Timer } from 'lucide-react';
import { cn } from '../lib/utils';
import { schedulerService } from '../services/api';

const WORKLOAD_PRESETS = [
  { name: 'Mixed', icon: Layers, data: [
    { arrival: 0, burst: 12, priority: 1, pid: 0, program: 'build' },
    { arrival: 2, burst: 4, priority: 2, pid: 1, program: 'shell' },
    { arrival: 5, burst: 8, priority: 1, pid: 2, program: 'build' },
    { arrival: 8, burst: 2, priority: 3, pid: 3, program: 'shell' },
  ]},
  { name: 'Compute Heavy', icon: Monitor, data: [
    { arrival: 0, burst: 40, priority: 1, pid: 0, program: 'render' },
    { arrival: 1, burst: 35, priority: 1, pid: 1, program: 'render' },
    { arrival: 2, burst: 45, priority: 1, pid: 2, program: 'render' },
  ]},
  { name: 'IO Intensive', icon: Zap, data: [
    { arrival: 0, burst: 2, priority: 1, pid: 0, program: 'shell' },
    { arrival: 1, burst: 3, priority: 1, pid: 1, program: 'grep' },
    { arrival: 2, burst: 2, priority: 1, pid: 2, program: 'shell' },
    { arrival: 3, burst: 4, priority: 1, pid: 3, program: 'grep' },
    { arrival: 4, burst: 2, priority: 1, pid: 4, program: 'shell' },
  ]}
];

// Periodic real-time task sets (period T, worst-case execution time C), expanded
// into jobs by the backend over two hyperperiods. Jobs must finish before the
// task's next release.
const RT_PRESETS = [
  {
    name: 'RT Schedulable',
    hint: 'U = 0.75, below the RMS bound (0.78): EDF and RMS both meet every deadline on 1 core',
    tasks: [
      { name: 'sensor', period: 8, wcet: 2 },
      { name: 'control', period: 12, wcet: 3 },
      { name: 'logger', period: 16, wcet: 4 },
    ],
  },
  {
    name: 'RT Overloaded',
    hint: 'U = 0.96, above the RMS bound: on 1 core EDF meets every deadline, RMS misses some',
    tasks: [
      { name: 'sensor', period: 6, wcet: 2 },
      { name: 'control', period: 8, wcet: 3 },
      { name: 'logger', period: 12, wcet: 3 },
    ],
  },
];

const TRACE_FORMATS = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'google', label: 'Google 2011' },
  { value: 'alibaba', label: 'Alibaba 2018' },
];

// Fresh row objects, so edits can never write back into WORKLOAD_PRESETS
const copyRows = (rows) => rows.map((p) => ({ ...p }));

// Lowest allowed value per field. A burst of 0 would never "complete" in the simulator.
const FIELD_MIN = { arrival: 0, burst: 1, priority: 0 };

export default function InputPanel({ onSimulate, isSimulating, onConfigChange }) {
  const [processes, setProcesses] = useState(() => copyRows(WORKLOAD_PRESETS[0].data));
  const [scheduler, setScheduler] = useState('HYBRID');
  const [cores, setCores] = useState(2);
  // quantum is the last valid value (>= 1); quantumText is what's in the box,
  // so the field can be cleared mid-edit without sending an invalid value.
  const [quantum, setQuantum] = useState(2);
  const [quantumText, setQuantumText] = useState('2');
  const [traceFormat, setTraceFormat] = useState('auto');
  const [rtBusy, setRtBusy] = useState(false);
  // { type: 'info' | 'error', text } — feedback for the real-time tools
  const [rtStatus, setRtStatus] = useState(null);

  const handleQuantumChange = (value) => {
    setQuantumText(value);
    const parsed = Number(value);
    if (Number.isInteger(parsed) && parsed >= 1) {
      setQuantum(parsed);
    }
  };

  // Notify parent when anything the comparison depends on changes. The selected
  // scheduler is deliberately excluded: /compare runs every scheduler anyway.
  useEffect(() => {
    if (onConfigChange) {
      onConfigChange({ num_cores: cores, quantum, processes });
    }
  }, [cores, processes, quantum, onConfigChange]);

  const addProcess = () => {
    const lastPid = processes.length > 0 ? processes[processes.length - 1].pid : -1;
    setProcesses([...processes, { 
      pid: lastPid + 1, 
      arrival: 0, 
      burst: 5, 
      priority: 1,
      program: '',
    }]);
  };

  const updateProcess = (index, field, value) => {
    const newValue = parseInt(value, 10);
    if (isNaN(newValue)) return;

    const clamped = Math.max(FIELD_MIN[field] ?? 0, newValue);
    // Replace the edited row with a new object instead of mutating it in place
    setProcesses(processes.map((p, i) => (i === index ? { ...p, [field]: clamped } : p)));
  };

  // Program is a free-text label (executable / user); blank means unknown
  const updateProgram = (index, value) => {
    setProcesses(processes.map((p, i) => (i === index ? { ...p, program: value.slice(0, 24) } : p)));
  };

  // Deadline is optional (blank = none) and must come after the arrival
  const updateDeadline = (index, value) => {
    setProcesses(processes.map((p, i) => {
      if (i !== index) return p;
      if (value === '') return { ...p, deadline: null };
      const v = parseInt(value, 10);
      return isNaN(v) ? p : { ...p, deadline: Math.max(p.arrival + 1, v) };
    }));
  };

  // Runs an API call that returns { processes }, loads them into the editor
  const runRtAction = async (action, describe) => {
    setRtBusy(true);
    setRtStatus(null);
    try {
      const res = await action();
      setProcesses(res.data.processes.map((p) => ({ ...p, program: p.program ?? '' })));
      setRtStatus({ type: 'info', text: describe(res.data) });
    } catch (err) {
      setRtStatus({ type: 'error', text: err.response?.data?.error || 'Request failed — is the backend running?' });
    } finally {
      setRtBusy(false);
    }
  };

  const applyRtPreset = (preset) => runRtAction(
    () => schedulerService.periodicWorkload(preset.tasks, cores),
    (d) => `${d.processes.length} jobs over ${d.horizon} ticks · U = ${d.utilisation.toFixed(2)} (RMS bound ${d.rms_bound.toFixed(2)})`,
  );

  const autoDeadlines = () => runRtAction(
    () => schedulerService.addDeadlines(processes, 3),
    () => 'Deadlines added where missing: arrival + 3 × burst',
  );

  const handleTraceUpload = (event) => {
    const file = event.target.files[0];
    event.target.value = null;
    if (!file) return;
    runRtAction(
      () => schedulerService.importTrace(file, { format: traceFormat, max_jobs: 60, load: 0.9, num_cores: cores }),
      (d) => `Imported ${d.stats.imported} of ${d.stats.completed_tasks_in_file} tasks (${d.format}) · `
        + `${d.stats.seconds_per_tick.toFixed(1)} s/tick · ${d.stats.with_deadline} with deadlines`,
    );
  };

  const removeProcess = (index) => {
    setProcesses(processes.filter((_, i) => i !== index));
  };

  const applyPreset = (data) => {
    setProcesses(copyRows(data));
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').map(l => l.trim()).filter(l => l);
        if (lines.length < 2) return; // Need header + data
        
        const headers = lines[0].toLowerCase().split(',').map(h => h.trim());
        const findCol = (...tests) => {
          for (const test of tests) {
            const idx = headers.findIndex(test);
            if (idx >= 0) return idx;
          }
          return -1;
        };
        const pidIdx = findCol(h => h === 'pid' || h === 'id', h => h.includes('pid'));
        const arrivalIdx = findCol(h => h.includes('arrival') || h.includes('arrive'));
        // Prefer an explicit "burst" column; "time" alone would also match "arrival_time"
        const burstIdx = findCol(h => h.includes('burst'), h => h.includes('time') && !h.includes('arriv'));
        const priorityIdx = findCol(h => h.includes('priority') || h.includes('prio'));
        const programIdx = findCol(h => h.includes('program') || h.includes('command') || h === 'cmd' || h.includes('exe'));
        const deadlineIdx = findCol(h => h.includes('deadline') || h === 'due');

        const newProcesses = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          if (values.length < 2) continue; // Skip incomplete lines

          const arrival = Math.max(FIELD_MIN.arrival, arrivalIdx >= 0 ? parseInt(values[arrivalIdx]) || 0 : 0);
          const deadline = deadlineIdx >= 0 ? parseInt(values[deadlineIdx]) : NaN;
          newProcesses.push({
            pid: pidIdx >= 0 ? parseInt(values[pidIdx]) : i - 1,
            arrival,
            burst: Math.max(FIELD_MIN.burst, burstIdx >= 0 ? parseInt(values[burstIdx]) || 5 : 5),
            priority: Math.max(FIELD_MIN.priority, priorityIdx >= 0 ? parseInt(values[priorityIdx]) || 1 : 1),
            program: programIdx >= 0 ? (values[programIdx] || '').slice(0, 24) : '',
            deadline: !isNaN(deadline) && deadline > arrival ? deadline : null,
          });
        }
        
        if (newProcesses.length > 0) {
          setProcesses(newProcesses);
        }
      } catch (err) {
        console.error("Failed to parse CSV", err);
      } finally {
        event.target.value = null; // reset input
      }
    };
    reader.readAsText(file);
  };
  return (
    <div className="flex flex-col gap-6">
      {/* Configuration Header */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-xs text-primary/80 uppercase font-bold tracking-widest pl-1">Scheduler Architecture</label>
          <select 
            value={scheduler}
            onChange={(e) => setScheduler(e.target.value)}
            className="glass-input text-mtx4 border-ink/10"
          >
            <option value="HYBRID">Hybrid (PPO + LSTM)</option>
            <option value="PPO">PPO Agent</option>
            <option value="FCFS">First-Come-First-Serve</option>
            <option value="SJF">Shortest Job First</option>
            <option value="RR">Round Robin</option>
            <option value="MLFQ">Multi-Level Feedback Queue</option>
            <option value="EDF">Earliest Deadline First (real-time)</option>
            <option value="RMS">Rate Monotonic (real-time)</option>
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between pl-1">
            <label className="text-xs text-primary/70 uppercase font-bold tracking-widest">CPU Cores</label>
          </div>
          <div className="flex gap-2">
            {[1, 2, 4, 8].map(c => (
              <button
                key={c}
                onClick={() => setCores(c)}
                className={cn(
                  "flex-1 py-2 rounded-lg border transition-all text-sm font-bold",
                  cores === c ? "bg-primary/20 border-primary text-primary" : "border-mtx1 text-mtx3 hover:border-mtx2"
                )}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Round Robin quantum — always visible: the comparison runs RR regardless of the selected scheduler */}
      <div className="flex items-center justify-between gap-4 pl-1">
        <div className="flex flex-col">
          <label htmlFor="rr-quantum" className="text-xs text-primary/70 uppercase font-bold tracking-widest">
            RR Time Quantum
          </label>
          <span className="text-[10px] text-mtx2 italic mt-0.5">
            {scheduler === 'RR' ? 'Ticks each process runs before preemption' : 'Used by Round Robin in the comparison'}
          </span>
        </div>
        <input
          id="rr-quantum"
          type="number"
          min={1}
          step={1}
          value={quantumText}
          onChange={(e) => handleQuantumChange(e.target.value)}
          onBlur={() => setQuantumText(String(quantum))}
          className={cn(
            "glass-input w-24 text-center font-mono text-mtx4 border-ink/10",
            scheduler === 'RR' && "border-primary text-primary"
          )}
        />
      </div>

      {/* Presets */}
      <div className="flex flex-wrap gap-2">
        {WORKLOAD_PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() => applyPreset(p.data)}
            className="flex-1 flex items-center justify-center gap-2 py-3 glass-card--inset-border glass-card hover:bg-ink/[0.06] transition-all group"
          >
            <span className="text-xs font-bold uppercase tracking-wider text-mtx4">{p.name}</span>
          </button>
        ))}
      </div>

      {/* Real-time tools */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 pl-1">
          <Timer size={12} className="text-rose" />
          <span className="text-xs text-rose/80 uppercase font-bold tracking-widest">Real-Time &amp; Traces</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {RT_PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => applyRtPreset(p)}
              disabled={rtBusy}
              title={p.hint}
              className="flex-1 min-w-[120px] py-2 glass-card--inset-border glass-card hover:bg-ink/[0.06] transition-all text-xs font-bold uppercase tracking-wider text-mtx4 disabled:opacity-50"
            >
              {p.name}
            </button>
          ))}
          <button
            onClick={autoDeadlines}
            disabled={rtBusy || processes.length === 0}
            title="Give every process without a deadline: deadline = arrival + 3 × burst"
            className="flex-1 min-w-[120px] py-2 flex items-center justify-center gap-1.5 glass-card--inset-border glass-card hover:bg-ink/[0.06] transition-all text-xs font-bold uppercase tracking-wider text-mtx4 disabled:opacity-50"
          >
            <AlarmClock size={14} /> Auto Deadlines
          </button>
        </div>
        <div className="flex gap-2 items-stretch">
          <select
            value={traceFormat}
            onChange={(e) => setTraceFormat(e.target.value)}
            aria-label="Trace format"
            className="glass-input w-36 shrink-0 text-xs text-mtx4 border-ink/10 py-2"
          >
            {TRACE_FORMATS.map((f) => (
              <option key={f.value} value={f.value}>{f.label}</option>
            ))}
          </select>
          <label
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg bg-rose/10 text-rose hover:bg-rose/20 transition-colors cursor-pointer text-xs font-bold uppercase tracking-wider whitespace-nowrap",
              rtBusy && "opacity-50 pointer-events-none"
            )}
            title="Import a Google cluster-data 2011 task_events or Alibaba 2018 batch_task CSV (first 60 completed tasks, rescaled to ~90% load)"
          >
            <FileUp size={14} /> Import Cluster Trace
            <input type="file" accept=".csv,.txt" className="hidden" onChange={handleTraceUpload} data-testid="trace-input" />
          </label>
        </div>
        {rtStatus && (
          <p className={cn("text-[11px] pl-1", rtStatus.type === 'error' ? 'text-red-400' : 'text-mtx3')} role="status">
            {rtStatus.text}
          </p>
        )}
      </div>

      {/* Process Table */}
      <div className="glass-card--inset-border glass-card overflow-hidden">
        <div className="p-4 border-b border-mtx1 flex items-center justify-between bg-ink/[0.02]">
          <h3 className="text-sm font-bold uppercase tracking-widest text-mtx3">Workload Definition</h3>
          <div className="flex gap-2">
            <label 
              className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-colors cursor-pointer flex items-center gap-2"
              title="Import from CSV"
            >
              <Upload size={16} />
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
            </label>
            <button 
              onClick={addProcess}
              className="p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
              title="Add Process"
            >
              <Plus size={16} />
            </button>
          </div>
        </div>
        
        <div className="max-h-[300px] overflow-x-auto overflow-y-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-[10px] text-mtx2 uppercase tracking-widest border-b border-mtx1">
              <tr>
                <th className="px-2 py-3">PID</th>
                <th className="px-2 py-3">Arrival</th>
                <th className="px-2 py-3">Burst</th>
                <th className="px-2 py-3">Priority</th>
                <th className="px-2 py-3" title="Optional program / executable label. Jobs of the same program tend to have similar bursts; the Hybrid scheduler learns from them.">Program</th>
                <th className="px-2 py-3" title="Optional absolute deadline (tick). Blank = no deadline.">Deadline</th>
                <th className="px-2 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {processes.map((p, i) => (
                <tr key={i} className="border-b border-mtx1 group hover:bg-ink/[0.04] transition-colors">
                  <td className="px-2 py-3 font-mono text-primary font-bold">P{p.pid}</td>
                  <td className="px-2 py-3">
                    <input 
                      type="number" 
                      value={p.arrival}
                      onChange={(e) => updateProcess(i, 'arrival', e.target.value)}
                      className="w-12 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-2 py-3">
                    <input 
                      type="number" 
                      value={p.burst}
                      onChange={(e) => updateProcess(i, 'burst', e.target.value)}
                      className="w-12 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-2 py-3">
                    <input 
                      type="number" 
                      value={p.priority}
                      onChange={(e) => updateProcess(i, 'priority', e.target.value)}
                      className="w-12 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-2 py-3">
                    <input
                      type="text"
                      value={p.program ?? ''}
                      placeholder="—"
                      onChange={(e) => updateProgram(i, e.target.value)}
                      className="w-16 bg-transparent border-none focus:ring-0 font-mono text-mtx5 placeholder:text-mtx2"
                    />
                    {p.period != null && (
                      <span className="ml-1 text-[9px] font-mono text-rose/80" title="Period of the periodic task this job belongs to (used by RMS)">
                        T={p.period}
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-3">
                    <input
                      type="number"
                      value={p.deadline ?? ''}
                      placeholder="—"
                      min={p.arrival + 1}
                      onChange={(e) => updateDeadline(i, e.target.value)}
                      className="w-12 bg-transparent border-none focus:ring-0 font-mono text-mtx5 placeholder:text-mtx2"
                    />
                  </td>
                  <td className="px-1 py-3 ">
                    <button 
                      onClick={() => removeProcess(i)}
                      className="p-2 text-mtx2 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <button
        onClick={() => onSimulate({ scheduler, num_cores: cores, quantum, processes })}
        disabled={isSimulating || processes.length === 0}
        className="btn-primary w-full h-14 flex items-center justify-center gap-2 group"
      >
        <Play size={20} className={cn("transition-transform text-white group-hover:scale-110", isSimulating && "animate-pulse")} />
        <span className="text-lg text-white font-bold">RUN SIMULATION</span>
      </button>
    </div>
  );
}
