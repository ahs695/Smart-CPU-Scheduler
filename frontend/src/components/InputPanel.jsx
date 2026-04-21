import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Play, Zap, Monitor, Layers, AlertCircle,Upload } from 'lucide-react';
import { cn } from '../lib/utils';

const WORKLOAD_PRESETS = [
  { name: 'Mixed', icon: Layers, data: [
    { arrival: 0, burst: 12, priority: 1, pid: 0 },
    { arrival: 2, burst: 4, priority: 2, pid: 1 },
    { arrival: 5, burst: 8, priority: 1, pid: 2 },
    { arrival: 8, burst: 2, priority: 3, pid: 3 },
  ]},
  { name: 'Compute Heavy', icon: Monitor, data: [
    { arrival: 0, burst: 40, priority: 1, pid: 0 },
    { arrival: 1, burst: 35, priority: 1, pid: 1 },
    { arrival: 2, burst: 45, priority: 1, pid: 2 },
  ]},
  { name: 'IO Intensive', icon: Zap, data: [
    { arrival: 0, burst: 2, priority: 1, pid: 0 },
    { arrival: 1, burst: 3, priority: 1, pid: 1 },
    { arrival: 2, burst: 2, priority: 1, pid: 2 },
    { arrival: 3, burst: 4, priority: 1, pid: 3 },
    { arrival: 4, burst: 2, priority: 1, pid: 4 },
  ]}
];

export default function InputPanel({ onSimulate, isSimulating }) {
  const [processes, setProcesses] = useState(WORKLOAD_PRESETS[0].data);
  const [scheduler, setScheduler] = useState('HYBRID');
  const [cores, setCores] = useState(2);

  const isAIScheduler = ['HYBRID', 'PPO'].includes(scheduler);

  // AI models are currently trained for fixed 2-core environments
  useEffect(() => {
    if (isAIScheduler) {
      setCores(2);
    }
  }, [scheduler]);

  const addProcess = () => {
    const lastPid = processes.length > 0 ? processes[processes.length - 1].pid : -1;
    setProcesses([...processes, { 
      pid: lastPid + 1, 
      arrival: 0, 
      burst: 5, 
      priority: 1 
    }]);
  };

  const updateProcess = (index, field, value) => {
    const updated = [...processes];
    updated[index][field] = parseInt(value) || 0;
    setProcesses(updated);
  };

  const removeProcess = (index) => {
    setProcesses(processes.filter((_, i) => i !== index));
  };

  const applyPreset = (data) => {
    setProcesses(data);
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
        const pidIdx = headers.findIndex(h => h.includes('pid') || h.includes('id'));
        const arrivalIdx = headers.findIndex(h => h.includes('arrival') || h.includes('arrive'));
        const burstIdx = headers.findIndex(h => h.includes('burst') || h.includes('time'));
        const priorityIdx = headers.findIndex(h => h.includes('priority') || h.includes('prio'));

        const newProcesses = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          if (values.length < 2) continue; // Skip incomplete lines

          newProcesses.push({
            pid: pidIdx >= 0 ? parseInt(values[pidIdx]) : i - 1,
            arrival: arrivalIdx >= 0 ? parseInt(values[arrivalIdx]) || 0 : 0,
            burst: burstIdx >= 0 ? parseInt(values[burstIdx]) || 5 : 5,
            priority: priorityIdx >= 0 ? parseInt(values[priorityIdx]) || 1 : 1,
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
          <label className="text-xs text-white text-center bg-accent rounded-3xl uppercase font-bold tracking-widest ">Scheduler Architecture</label>
          <select 
  value={scheduler}
  onChange={(e) => setScheduler(e.target.value)}
  className="glass-input text-black"
>
  <option value="HYBRID">Hybrid (PPO + LSTM)</option>
  <option value="PPO">PPO Agent</option>
  <option value="FCFS">First-Come-First-Serve</option>
  <option value="SJF">Shortest Job First</option>
  <option value="RR">Round Robin</option>
  <option value="MLFQ">Multi-Level Feedback Queue</option>
</select>
</div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between pl-1">
            <label className="text-xs text-primary/70 uppercase font-bold tracking-widest">CPU Cores</label>
            {isAIScheduler && (
              <span className="text-[10px] text-primary font-bold flex items-center gap-1 bg-primary/10 px-1 py-0.5 rounded-full">
                 Optimized for 2
              </span>
            )}
          </div>
          <div className="flex gap-2">
            {[1, 2, 4, 8].map(c => (
              <button
                key={c}
                disabled={isAIScheduler && c !== 2}
                onClick={() => setCores(c)}
                className={cn(
                  "flex-1 py-2 rounded-lg border transition-all text-sm font-bold",
                  cores === c ? "bg-primary/20 border-primary text-primary" : "border-mtx1 text-mtx3 hover:border-white",
                  isAIScheduler && c !== 2 && "opacity-20 grayscale cursor-not-allowed border-dashed"
                )}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Presets */}
      <div className="flex gap-2">
        {WORKLOAD_PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() => applyPreset(p.data)}
            className="flex-1 flex items-center justify-center gap-2 py-3 glass-card--inset-border glass-card hover:bg-gray-200 transition-all group"
          >
            <span className="text-xs font-bold uppercase tracking-wider text-mtx4">{p.name}</span>
          </button>
        ))}
      </div>

      {/* Process Table */}
      <div className="glass-card--inset-border glass-card overflow-hidden">
        <div className="p-4 border-b border-mtx1 flex items-center justify-between bg-block">
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
        
        <div className="max-h-[300px] overflow-x-hidden overflow-y-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-[10px] text-mtx2 uppercase tracking-widest border-b border-mtx1">
              <tr>
                <th className="px-4 py-3">PID</th>
                <th className="px-4 py-3">Arrival</th>
                <th className="px-4 py-3">Burst</th>
                <th className="px-4 py-3">Priority</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {processes.map((p, i) => (
                <tr key={i} className="border-b border-mtx1 group hover:bg-gray-200 transition-colors">
                  <td className="px-4 py-3 font-mono text-primary font-bold">P{p.pid}</td>
                  <td className="px-4 py-3">
                    <input 
                      type="number" 
                      value={p.arrival}
                      onChange={(e) => updateProcess(i, 'arrival', e.target.value)}
                      className="w-16 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <input 
                      type="number" 
                      value={p.burst}
                      onChange={(e) => updateProcess(i, 'burst', e.target.value)}
                      className="w-16 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <input 
                      type="number" 
                      value={p.priority}
                      onChange={(e) => updateProcess(i, 'priority', e.target.value)}
                      className="w-16 bg-transparent border-none focus:ring-0 font-mono text-mtx5"
                    />
                  </td>
                  <td className="px-4 py-3 text-right">
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
        onClick={() => onSimulate({ scheduler, num_cores: cores, processes })}
        disabled={isSimulating || processes.length === 0}
        className="btn-primary w-full h-14 flex items-center justify-center gap-2 group"
      >
        <Play size={20} className={cn("transition-transform text-white group-hover:scale-110", isSimulating && "animate-pulse")} />
        <span className="text-lg text-white font-bold">RUN SIMULATION</span>
      </button>
    </div>
  );
}
