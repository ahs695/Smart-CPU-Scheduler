import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, ZAxis 
} from 'recharts';

export function RewardCurve({ data }) {
  return (
    <div className="glass-card glass-card--inset-border p-6 h-[300px]">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80">RL Agent Learning Progress</h3>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-600" />
            <span className="text-[10px] text-mtx4 uppercase font-bold">Total Reward</span>
          </div>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height="80%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" vertical={false} />
          <XAxis 
            dataKey="episode" 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Training Episode', position: 'insideBottom', offset: -1, fill: '040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <YAxis 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Total Reward', angle: -90, position: 'insideLeft', offset: 10, fill: '#040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#141417', border: '1px solid #010101', borderRadius: '8px' }}
            itemStyle={{ color: '#3b82f6', fontSize: '12px' }}
          />
          <Area 
            type="monotone" 
            dataKey="reward" 
            stroke="#3b82f6" 
            fillOpacity={1} 
            fill="url(#colorReward)" 
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LSTMPredictionChart({ data }) {
  return (
    <div className="glass-card glass-card--inset-border p-6 h-[300px]">
      <h3 className="text-sm font-bold uppercase tracking-widest text-secondary/80 mb-6">LSTM Burst Prediction Accuracy</h3>
      
      <ResponsiveContainer width="100%" height="90%">
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" vertical={false} />
          <XAxis 
            dataKey="id" 
            name="Sample ID" 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Sample ID', position: 'insideBottom', offset: -1, fill: '#040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <YAxis 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Burst Length (ms)', angle: -90, position: 'insideLeft', offset: 8, fill: '#040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ backgroundColor: '#141417', border: '1px solid #010101', borderRadius: '8px' }}
          />
          <Scatter name="Actual" data={data} fill="#225599" shape="circle" dataKey="actual" />
          <Scatter name="Predicted" data={data} fill="#10b981" shape="cross" dataKey="predicted" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ComparisonChart({ data }) {
  const chartData = Array.isArray(data) ? data : [];
  
  if (chartData.length === 0) {
    return (
      <div className="glass-card glass-card--inset-border p-6 h-[400px]">
        <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80 mb-6">Architecture performance benchmark</h3>
        <div className="flex flex-col items-center justify-center h-[80%] opacity-50">
          <p className="text-sm font-medium">No Data Available</p>
          <p className="text-xs text-mtx3 italic mt-2">Run a simulation to see performance comparisons here.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--inset-border p-6 h-[400px]">
      <h3 className="text-sm font-bold uppercase tracking-widest text-primary/80 mb-6">Architecture performance benchmark</h3>
      
      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" vertical={false} />
          <XAxis 
            dataKey="name" 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Scheduling Algorithm', position: 'insideBottom', offset: -1, fill: '#040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <YAxis 
            stroke="#020202" 
            fontSize={10} 
            tickLine={false} 
            axisLine={false}
            label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', offset: 10, fill: '#040404', fontSize: 10, fontWeight: 'bold' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#141417', border: '1px solid #010101', borderRadius: '8px' }}
          />
          <Line type="monotone" dataKey="waiting" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: '#3b82f6' }} strokeDasharray="5 5" name="Wait Time" />
          <Line type="monotone" dataKey="turnaround" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4, fill: '#8b5cf6' }} name="Turnaround" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
