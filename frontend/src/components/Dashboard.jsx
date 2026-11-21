import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { schedulerService } from '../services/api';
import InputPanel from './InputPanel';
import GanttChart from './GanttChart';
import SLAAlerts from './SLAAlerts';
import '../App.css';
import MetricsPanel from './MetricsPanel';
import { LSTMPredictionChart, ComparisonChart } from './TrainingCharts';
import { Activity, BarChart3, Info, AlertCircle, Cpu, Sparkles, Sun, Moon } from 'lucide-react';
import { useTheme } from '../lib/theme';

// Wait this long after the last workload/cores/quantum edit before re-running /compare
const COMPARE_DEBOUNCE_MS = 400;

export default function Dashboard() {
  const [simulationData, setSimulationData] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [comparisonData, setComparisonData] = useState([]);
  const [isComparing, setIsComparing] = useState(false);
  const [compareCores, setCompareCores] = useState(2);
  // { status: 'loading' | 'ok' | 'unavailable', samples, mae, baseline_mae }
  const [lstmEval, setLstmEval] = useState({ status: 'loading' });
  const [backendStatus, setBackendStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [lastSimConfig, setLastSimConfig] = useState(null);
  // Bumped per run: remounts the Gantt chart so playback (and SLA alerts) restart
  const [runId, setRunId] = useState(0);
  const [playbackTime, setPlaybackTime] = useState(-1);
  const { theme, toggle: toggleTheme } = useTheme();

  const compareTimerRef = useRef(null);
  const compareAbortRef = useRef(null);

  // Only the most recent /compare request is allowed to update state:
  // starting a new one aborts the previous in-flight request.
  const fetchComparison = useCallback(async (config) => {
    compareAbortRef.current?.abort();
    const controller = new AbortController();
    compareAbortRef.current = controller;

    setIsComparing(true);
    try {
      const comp = await schedulerService.compareProcesses({
        processes: config.processes,
        num_cores: config.num_cores,
        quantum: config.quantum,
      }, { signal: controller.signal });
      const raw = comp.data;
      if (raw && raw.comparison) {
        const arr = Object.entries(raw.comparison)
          .filter(([, v]) => v.metrics)
          .map(([name, v]) => ({
            name,
            waiting:    v.metrics.avg_waiting_time    ?? 0,
            turnaround: v.metrics.avg_turnaround_time ?? 0,
            fairness:   v.metrics.fairness_index      ?? 0,
            missRate:   v.metrics.deadline_miss_rate,     // null when no deadlines
          }));
        setComparisonData(arr);
        setCompareCores(config.num_cores);
      }
    } catch (err) {
      if (axios.isCancel(err)) return; // superseded by a newer request
      console.error("Failed to fetch comparison data", err);
    } finally {
      if (compareAbortRef.current === controller) {
        compareAbortRef.current = null;
        setIsComparing(false);
      }
    }
  }, []);

  // Called by InputPanel on mount and whenever workload, cores or quantum change.
  const handleConfigChange = useCallback((config) => {
    clearTimeout(compareTimerRef.current);
    compareTimerRef.current = setTimeout(() => fetchComparison(config), COMPARE_DEBOUNCE_MS);
  }, [fetchComparison]);

  useEffect(() => () => {
    clearTimeout(compareTimerRef.current);
    compareAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      const [status, lstm] = await Promise.allSettled([
        schedulerService.checkStatus(),
        schedulerService.getLSTMPredictions(),
      ]);
      setBackendStatus(status.status === 'fulfilled' ? 'online' : 'offline');
      if (lstm.status === 'fulfilled') {
        setLstmEval({ status: 'ok', ...lstm.value.data });
      } else {
        console.error("Failed to fetch LSTM predictions", lstm.reason);
        setLstmEval({ status: 'unavailable' });
      }
    };
    fetchData();
  }, []);

  const handleSimulate = async (config) => {
    setIsSimulating(true);
    setError(null);
    try {
      const response = await schedulerService.simulate(config);
      setSimulationData(response.data);
      setRunId((r) => r + 1);
      setPlaybackTime(-1);
      setLastSimConfig(config);
    } catch (err) {
      setError(err.response?.data?.error || "Simulation failed. Please check if the backend is running.");
      console.error(err);
    } finally {
      setIsSimulating(false);
    }
  };

  // pid -> deadline, and the pids that missed, for the Gantt chart
  const deadlineInfo = useMemo(() => {
    const events = simulationData?.deadline_events || [];
    return {
      deadlines: Object.fromEntries(events.map((e) => [e.pid, e.deadline])),
      missed: new Set(events.filter((e) => e.missed).map((e) => e.pid)),
    };
  }, [simulationData]);
  const hasDeadlines = comparisonData.some((d) => d.missRate != null);

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col gap-8 max-w-[1600px] mx-auto relative">
      {/* Ambient background orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-primary/[0.04] blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-secondary/[0.04] blur-[120px]" />
        <div className="absolute top-[40%] left-[50%] w-[300px] h-[300px] rounded-full bg-accent/[0.03] blur-[100px]" />
      </div>

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="logo-title text-3xl font-black tracking-tighter flex items-center gap-3">
            <Sparkles className="text-primary" size={28} />
            AIMOS <span className="text-mtx3 font-light tracking-normal"> Dashboard</span>
          </h1>
          <p className="text-mtx2 text-sm mt-1 font-medium italic">Research-grade AI-Driven Multi Core CPU scheduling simulation</p>
        </div>

        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-[10px] font-bold uppercase tracking-widest border backdrop-blur-sm ${
            backendStatus === 'online'
              ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
              : 'bg-red-500/20 text-red-400 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.15)]'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              backendStatus === 'online'
                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse'
                : 'bg-red-400 shadow-[0_0_8px_rgba(239,68,68,0.8)]'
            }`} />
            Backend {backendStatus}
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            data-testid="theme-toggle"
            className="relative w-10 h-10 grid place-items-center rounded-full border border-mtx1 bg-card backdrop-blur-sm text-mtx3 hover:text-primary hover:border-primary/40 transition-colors overflow-hidden"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={theme}
                initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                animate={{ rotate: 0, opacity: 1, scale: 1 }}
                exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                transition={{ duration: 0.2 }}
                className="grid place-items-center"
              >
                {theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}
              </motion.span>
            </AnimatePresence>
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl flex items-center gap-3 backdrop-blur-sm">
          <AlertCircle size={20} />
          <span className="text-sm font-bold">{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* Left Side: Input & Settings */}
        <div className="xl:col-span-4 flex flex-col gap-8">
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Cpu size={14} className="text-primary" />
              <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-primary/80">Simulation Control</h2>
            </div>
            <InputPanel 
              onSimulate={handleSimulate} 
              isSimulating={isSimulating}
              onConfigChange={handleConfigChange}
            />
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 size={14} className="text-secondary" />
              <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-secondary/80">Analytics & Comparison</h2>
            </div>
            <ComparisonChart 
              data={comparisonData} 
              numCores={compareCores}
              isLoading={isComparing} 
            />
          </section>
        </div>

        {/* Right Side: Live View & Training */}
        <div className="xl:col-span-8 flex flex-col gap-8">
          <section>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Activity size={14} className="text-accent" />
                <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-accent/80">Live Execution Visualization</h2>
              </div>
              {simulationData && lastSimConfig && (
                <div className="text-[10px] font-bold bg-primary/10 text-primary border border-primary/20 px-3 py-1 rounded-full uppercase tracking-tighter shadow-[0_0_10px_rgba(6,182,212,0.15)]">
                  {lastSimConfig.scheduler}
                  {lastSimConfig.scheduler === 'RR' && ` (q=${lastSimConfig.quantum})`}
                  {' · '}{simulationData.metrics.total_processes} PROCS
                  {' · '}{lastSimConfig.num_cores} {lastSimConfig.num_cores === 1 ? 'CORE' : 'CORES'}
                </div>
              )}
            </div>
            
            {simulationData ? (
              <div className="flex flex-col gap-6">
                <GanttChart
                  key={runId}
                  trace={simulationData.trace}
                  numCores={simulationData.trace[0].cores.length}
                  onTimeChange={setPlaybackTime}
                  deadlines={deadlineInfo.deadlines}
                  missedPids={deadlineInfo.missed}
                />
                <SLAAlerts
                  events={simulationData.deadline_events}
                  currentTime={playbackTime}
                  traceLength={simulationData.trace.length}
                  realtime={simulationData.realtime}
                />
                <MetricsPanel metrics={simulationData.metrics} />
                
                <div className="flex flex-col gap-4 mt-2">
                    <div className="overflow-hidden glass-card--inset-border glass-card p-6 border border-ink/[0.06] rounded-xl">
                      <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead>
                          <tr className="border-b border-ink/[0.08] text-mtx2">
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs">Algorithm</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-right">Avg Waiting</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-right">Avg Turnaround</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-center">Fairness Index</th>
                            {hasDeadlines && (
                              <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-right">Deadline Miss</th>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonData.map((data) => (
                            <tr key={data.name} className="border-b border-ink/[0.04] hover:bg-ink/[0.03] transition-colors">
                              <td className="py-4 px-4 font-bold text-primary">{data.name}</td>
                              <td className="py-4 px-4 text-right font-mono text-mtx4">{data.waiting.toFixed(1)}</td>
                              <td className="py-4 px-4 text-right font-mono text-mtx4">{data.turnaround.toFixed(1)}</td>
                              <td className="py-4 px-4 text-center font-mono">
                                <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                                  data.fairness > 0.85 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                    : data.fairness > 0.75 ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                                }`}>
                                  {data.fairness.toFixed(2)}
                                </span>
                              </td>
                              {hasDeadlines && (
                                <td className={`py-4 px-4 text-right font-mono ${data.missRate > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                  {data.missRate == null ? '—' : `${(data.missRate * 100).toFixed(1)}%`}
                                </td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                </div>
              </div>
            ) : (
              <div className="h-[400px] glass-card--inset-border glass-card flex flex-col items-center justify-center border-dashed border-ink/[0.08] opacity-60">
                <Cpu size={40} className="text-primary/30 mb-4" />
                <p className="text-sm font-medium text-mtx3">Ready to Simulate</p>
                <p className="text-xs text-mtx2 italic mt-1">Define workload inputs on the left to begin.</p>
              </div>
            )}
          </section>

          <LSTMPredictionChart evaluation={lstmEval} />

{/* How to Use + AI Insights row */}

  {/* HOW TO USE GUIDE */}
  <div className="glass-card glass-card--inset-border p-6 flex flex-col gap-4">
    <div className="flex items-center gap-3 border-b border-ink/[0.06] pb-3">
      <Info className="text-primary" size={18} />
      <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80">How to Use</h3>
    </div>
    <div className="flex flex-col gap-3 text-xs text-mtx3 leading-relaxed">
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-primary">01</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Configure Workload</h4>
          <p className="mt-1">Pick a preset or manually define processes with Arrival, Burst, and Priority. You can also import a <span className="text-primary font-semibold">.csv</span> file via the upload button.</p>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-secondary/10 border border-secondary/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-secondary">02</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Pick a Scheduler</h4>
          <p className="mt-1">Use <span className="text-secondary font-semibold">Hybrid</span> or <span className="text-secondary font-semibold">PPO</span> for AI-driven scheduling (trained models for 1, 2, 4 and 8 cores), or benchmark against FCFS, SJF, RR, or MLFQ.</p>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-accent">03</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Read the Gantt Chart</h4>
          <p className="mt-1">The Live Execution panel animates per-core execution after each run. Use <span className="text-accent font-semibold">1x–10x</span> playback speed to inspect or fast-forward scheduling decisions.</p>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-rose/10 border border-rose/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-rose">04</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Interpret Metrics</h4>
          <p className="mt-1"><span className="font-semibold text-mtx4">Avg Waiting</span> — idle queue time. <span className="font-semibold text-mtx4">Turnaround</span> — total completion time. <span className="font-semibold text-mtx4">Fairness Index</span> — how evenly CPU time is shared (1.0 = perfect).</p>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-primary">05</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Compare All Algorithms</h4>
          <p className="mt-1">The benchmark chart (under Simulation Control) and the comparison table re-run every scheduler whenever you change the workload, cores or quantum — use them to justify algorithm choice for your workload type.</p>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-rose/10 border border-rose/20 flex items-center justify-center shrink-0">
          <span className="text-xs font-bold text-rose">06</span>
        </div>
        <div>
          <h4 className="text-sm font-bold text-mtx4">Deadlines &amp; SLA Alerts</h4>
          <p className="mt-1">Load an <span className="text-rose font-semibold">RT</span> preset (periodic tasks), press <span className="font-semibold text-mtx4">Auto Deadlines</span>, type deadlines, or import a Google / Alibaba cluster trace. Run <span className="text-rose font-semibold">EDF</span> or <span className="text-rose font-semibold">RMS</span> and play the Gantt chart: each missed deadline raises an alert as it happens, and missed jobs get a red outline. Try <span className="font-semibold text-mtx4">RT Overloaded</span> on 1 core: EDF meets every deadline, RMS does not.</p>
        </div>
      </div>
    </div>
  </div>

  

          {/* AI Insights Panel (Placeholder logic for demonstration) */}
          
        </div>
      </div>
    </div>
  );
}
