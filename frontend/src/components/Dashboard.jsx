import React, { useState, useEffect } from 'react';
import { schedulerService } from '../services/api';
import InputPanel from './InputPanel';
import GanttChart from './GanttChart';
import MetricsPanel from './MetricsPanel';
import { RewardCurve, LSTMPredictionChart, ComparisonChart } from './TrainingCharts';
import { Activity, Brain, BarChart3, Info, AlertCircle } from 'lucide-react';

export default function Dashboard() {
  const [simulationData, setSimulationData] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [comparisonData, setComparisonData] = useState([]);
  const [isComparing, setIsComparing] = useState(false);
  const [rewardData, setRewardData] = useState([]);
  const [lstmSamples, setLstmSamples] = useState([]);
  const [backendStatus, setBackendStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [showMetricsTable, setShowMetricsTable] = useState(false);
  const [lastSimConfig, setLastSimConfig] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [rew, lstm, status] = await Promise.all([
          schedulerService.getRewardCurve(),
          schedulerService.getLSTMPredictions(),
          schedulerService.checkStatus()
        ]);
        setRewardData(rew.data);
        setLstmSamples(lstm.data);
        setBackendStatus('online');
      } catch (err) {
        console.error("Failed to fetch initial data", err);
        setBackendStatus('offline');
      }
    };
    fetchData();
  }, []);

  const fetchComparison = async (config) => {
    setIsComparing(true);
    try {
      const comp = await schedulerService.compareProcesses({
        processes: config.processes,
        num_cores: config.num_cores,
      });
      const raw = comp.data;
      if (raw && raw.comparison) {
        const arr = Object.entries(raw.comparison)
          .filter(([, v]) => v.metrics)
          .map(([name, v]) => ({
            name,
            waiting:    v.metrics.avg_waiting_time    ?? 0,
            turnaround: v.metrics.avg_turnaround_time ?? 0,
            fairness:   v.metrics.fairness_index      ?? 0,
          }));
        setComparisonData(arr);
      }
    } catch (err) {
      console.error("Failed to fetch comparison data", err);
    } finally {
      setIsComparing(false);
    }
  };

  const handleSimulate = async (config) => {
    setIsSimulating(true);
    setError(null);
    try {
      const response = await schedulerService.simulate(config);
      setSimulationData(response.data);
      setLastSimConfig(config);
      // Fetch fresh comparison for the same processes
      fetchComparison(config);
    } catch (err) {
      setError(err.response?.data?.error || "Simulation failed. Please check if the backend is running.");
      console.error(err);
    } finally {
      setIsSimulating(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col gap-8 max-w-[1600px] mx-auto">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tighter flex items-center gap-3">
            <div className="p-2 bg-primary rounded-xl shadow-[0_0_20px_rgba(59,130,246,0.3)]">
              <Brain size={28} className="text-white" />
            </div>
            AI SMART SCHEDULER <span className="text-white/30 font-light tracking-normal">DASHBOARD</span>
          </h1>
          <p className="text-white/60 text-sm mt-1 font-medium italic">Research-grade multi-core CPU scheduling simulation with LSTM + PPO Hybrid architecture.</p>
        </div>

        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-[10px] font-bold uppercase tracking-widest border ${
            backendStatus === 'online' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'
          }`}>
            <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`} />
            Backend {backendStatus}
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-4 rounded-xl flex items-center gap-3 animate-in fade-in slide-in-from-top-4">
          <AlertCircle size={20} />
          <span className="text-sm font-bold">{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* Left Side: Input & Settings */}
        <div className="xl:col-span-4 flex flex-col gap-8">
          <section>
            <div className="flex items-center gap-2 mb-4 text-primary/80">
              <Activity size={16} />
              <h2 className="text-xs font-bold uppercase tracking-[0.2em]">Simulation Control</h2>
            </div>
            <InputPanel 
              onSimulate={handleSimulate} 
              isSimulating={isSimulating}
            />
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4 text-primary/80">
              <BarChart3 size={16} />
              <h2 className="text-xs font-bold uppercase tracking-[0.2em]">Analytics & Comparison</h2>
            </div>
            <ComparisonChart data={comparisonData} />
          </section>
        </div>

        {/* Right Side: Live View & Training */}
        <div className="xl:col-span-8 flex flex-col gap-8">
          <section>
            <div className="flex items-center justify-between mb-4 text-primary/80">
              <div className="flex items-center gap-2">
                <Brain size={16} />
                <h2 className="text-xs font-bold uppercase tracking-[0.2em]">Live Execution Visualization</h2>
              </div>
              {simulationData && (
                <div className="text-[10px] font-bold bg-primary/10 text-primary border border-primary/20 px-3 py-1 rounded-full uppercase tracking-tighter">
                  ALGO: {simulationData.metrics.total_processes} PROCS
                </div>
              )}
            </div>
            
            {simulationData ? (
              <div className="flex flex-col gap-6">
                <GanttChart 
                  trace={simulationData.trace} 
                  numCores={simulationData.trace[0].cores.length}
                />
                <MetricsPanel metrics={simulationData.metrics} />
                
                <div className="flex flex-col gap-4 mt-2">
                  <button 
                    onClick={() => {
                      setShowMetricsTable(!showMetricsTable);
                      if (!showMetricsTable && comparisonData.length === 0 && lastSimConfig) {
                        fetchComparison(lastSimConfig);
                      }
                    }}
                    disabled={isComparing}
                    className="w-full sm:w-auto px-6 py-3 rounded-xl font-bold uppercase tracking-widest text-xs transition-all duration-300 border border-secondary/30 bg-secondary/10 text-secondary hover:bg-secondary/20 self-center disabled:opacity-50"
                  >
                    {isComparing ? "Comparing All Algorithms..." : showMetricsTable ? "Hide Metrics Comparison" : "Compare All Algorithms"}
                  </button>
                  {showMetricsTable && (
                    <div className="glass-card p-6 border border-white/10 rounded-xl overflow-x-auto animate-in fade-in slide-in-from-top-4">
                      <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead>
                          <tr className="border-b border-white/10 text-white/50">
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs">Algorithm</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-right">Avg Waiting</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-right">Avg Turnaround</th>
                            <th className="pb-3 px-4 font-medium uppercase tracking-wider text-xs text-center">Fairness Index</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonData.map((data, index) => (
                            <tr key={data.name} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                              <td className="py-4 px-4 font-bold text-primary">{data.name}</td>
                              <td className="py-4 px-4 text-right font-mono text-white/80">{data.waiting.toFixed(1)}</td>
                              <td className="py-4 px-4 text-right font-mono text-white/80">{data.turnaround.toFixed(1)}</td>
                              <td className="py-4 px-4 text-center font-mono">
                                <span className={`px-2 py-1 rounded-full text-xs font-bold ${data.fairness > 0.9 ? 'bg-emerald-500/10 text-emerald-400' : data.fairness > 0.8 ? 'bg-yellow-500/10 text-yellow-400' : 'bg-red-500/10 text-red-400'}`}>
                                  {data.fairness.toFixed(2)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-[400px] glass-card flex flex-col items-center justify-center border-dashed border-white/10 opacity-50">
                <Info size={48} className="text-white/10 mb-4" />
                <p className="text-sm font-medium">Ready to Simulate</p>
                <p className="text-xs text-white/30 italic">Define workload inputs on the left to begin.</p>
              </div>
            )}
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <RewardCurve data={rewardData} />
            <LSTMPredictionChart data={lstmSamples} />
          </div>

          {/* AI Insights Panel (Placeholder logic for demonstration) */}
          <section className="glass-card p-6 border-l-4 border-primary bg-primary/[0.02]">
            <div className="flex items-center gap-3 mb-4">
              <Brain className="text-primary" />
              <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80">Hybrid AI Insights</h3>
            </div>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                  <span className="text-xs font-bold text-primary">01</span>
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white/90 underline decoration-primary/30 underline-offset-4">Proactive SJF Preemption</h4>
                  <p className="text-xs text-white/70 leading-relaxed mt-1">
                    The Hybrid model utilized LSTM burst predictions to preempt Core 0 when a shorter task (P3) arrived, reducing potential waiting time by 14%.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-secondary/20 flex items-center justify-center shrink-0">
                  <span className="text-xs font-bold text-secondary">02</span>
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white/90 underline decoration-secondary/30 underline-offset-4">Load Balancing Efficiency</h4>
                  <p className="text-xs text-white/70 leading-relaxed mt-1">
                    PPO-guided scheduling distributed compute-intensive tasks across 2 cores to maintain a fairness index above 0.92, outperforming naive Round Robin.
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
