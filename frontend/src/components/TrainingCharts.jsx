import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter
} from 'recharts';
import { useChartColors } from '../lib/theme';

/**
 * Real LSTM burst predictions vs ground truth, served by /lstm-predictions.
 * evaluation: { status: 'loading' | 'ok' | 'unavailable', samples, mae, baseline_mae }
 */
export function LSTMPredictionChart({ evaluation }) {
  const { status = 'loading', samples = [], mae, baseline_mae } = evaluation || {};
  const hasData = status === 'ok' && samples.length > 0;
  const c = useChartColors();

  return (
    <div className="glass-card glass-card--inset-border p-6 h-[320px]">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-4">
        <h3 className="text-sm font-bold uppercase tracking-widest text-secondary/80">LSTM Burst Prediction Accuracy</h3>
        {hasData && (
          <span className="text-[10px] font-mono text-mtx3">
            MAE <span className="text-secondary font-bold">{mae.toFixed(2)}</span>
            {' '}vs last-burst baseline <span className="text-mtx4 font-bold">{baseline_mae.toFixed(2)}</span> ticks
            {' · '}{samples.length} held-out sequences
          </span>
        )}
      </div>

      {hasData ? (
        <ResponsiveContainer width="100%" height="85%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={c.grid} vertical={false} />
            <XAxis
              dataKey="id"
              name="Sample"
              type="number"
              stroke={c.axis}
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tick={{ fill: c.tick }}
              label={{ value: 'Sample', position: 'insideBottom', offset: -1, fill: c.label, fontSize: 10, fontWeight: 'bold' }}
            />
            <YAxis
              stroke={c.axis}
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tick={{ fill: c.tick }}
              label={{ value: 'Burst Length (ticks)', angle: -90, position: 'insideLeft', offset: 8, fill: c.label, fontSize: 10, fontWeight: 'bold' }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : v)}
              contentStyle={c.tooltip}
            />
            <Scatter name="Actual" data={samples} fill={c.actual} shape="circle" dataKey="actual" />
            <Scatter name="Predicted" data={samples} fill={c.predicted} shape="cross" dataKey="predicted" />
          </ScatterChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex flex-col items-center justify-center h-[80%] opacity-60">
          <p className="text-sm font-medium text-mtx4">
            {status === 'loading' ? 'Loading predictions…' : 'LSTM model not available'}
          </p>
          {status !== 'loading' && (
            <p className="text-xs text-mtx3 italic mt-2">Train it with python -m backend.ml.train_lstm and restart the backend.</p>
          )}
        </div>
      )}
    </div>
  );
}

export function ComparisonChart({ data, numCores = 2, isLoading = false }) {
  const chartData = Array.isArray(data) ? data : [];
  const c = useChartColors();
  
  if (chartData.length === 0) {
    return (
      <div className="glass-card glass-card--inset-border p-6 h-[400px]">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80">
            Architecture performance benchmark ({numCores} {numCores === 1 ? 'Core' : 'Cores'})
          </h3>
          {isLoading && (
            <span className="text-[10px] text-primary font-mono animate-pulse">Running comparison...</span>
          )}
        </div>
        <div className="flex flex-col items-center justify-center h-[80%] opacity-50">
          <p className="text-sm font-medium text-mtx4">{isLoading ? 'Evaluating schedulers...' : 'No Data Available'}</p>
          <p className="text-xs text-mtx3 italic mt-2">
            {isLoading ? `Running all algorithms across ${numCores} cores.` : 'Comparison unavailable — check that the backend is running.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--inset-border p-6 h-[400px]">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80">
            Architecture performance benchmark ({numCores} {numCores === 1 ? 'Core' : 'Cores'})
          </h3>
          {isLoading && (
            <span className="text-[10px] text-cyan-400 font-mono animate-pulse">updating...</span>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs font-semibold">
          <div className="flex items-center gap-1.5 text-mtx3">
            <span className="inline-block w-3 h-0.5 border-t-2 border-dashed" style={{ borderColor: c.waiting }}></span>
            <span>Wait Time</span>
          </div>
          <div className="flex items-center gap-1.5 text-mtx3">
            <span className="inline-block w-3 h-0.5 " style={{ backgroundColor: c.turnaround }}></span>
            <span>Turnaround</span>
          </div>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke={c.grid} vertical={false} />
          <XAxis 
            dataKey="name" 
            stroke={c.axis} 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            tick={{ fill: c.tick }}
            label={{ value: 'Scheduling Algorithm', position: 'insideBottom', offset: -1, fill: c.label, fontSize: 10, fontWeight: 'bold' }}
          />
          <YAxis 
            stroke={c.axis} 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            tick={{ fill: c.tick }}
            label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', offset: 10, fill: c.label, fontSize: 10, fontWeight: 'bold' }}
          />
          <Tooltip 
            cursor={{ stroke: c.axis, strokeDasharray: '3 3' }}
            contentStyle={c.tooltip}
          />
          <Line type="monotone" dataKey="waiting" stroke={c.waiting} strokeWidth={3} dot={{ r: 4, fill: c.waiting }} strokeDasharray="5 5" name="Wait Time" />
          <Line type="monotone" dataKey="turnaround" stroke={c.turnaround} strokeWidth={3} dot={{ r: 4, fill: c.turnaround }} name="Turnaround" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
