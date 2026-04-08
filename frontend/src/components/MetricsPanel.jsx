import React from 'react';
import { Clock, Activity, Zap, Cpu, ArrowRightLeft, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

const METRIC_CONFIG = {
  avg_waiting_time: { label: 'Avg Waiting', icon: Clock, color: 'text-blue-400', unit: 'ms' },
  avg_turnaround_time: { label: 'Turnaround', icon: Activity, color: 'text-purple-400', unit: 'ms' },
  throughput: { label: 'Throughput', icon: Zap, color: 'text-yellow-400', unit: 'p/ms' },
  fairness_index: { label: 'Fairness', icon: ShieldCheck, color: 'text-emerald-400', unit: '' },
  context_switches: { label: 'C-Switches', icon: ArrowRightLeft, color: 'text-orange-400', unit: '' },
  cpu_utilization: { label: 'Utilization', icon: Cpu, color: 'text-cyan-400', unit: '%' }
};

export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {Object.entries(METRIC_CONFIG).map(([key, config], idx) => {
        let value = metrics[key];
        if (value === undefined || value === null) value = 0;
        
        // Formatting
        const displayValue = typeof value === 'number' 
          ? (key === 'cpu_utilization' ? (value * 100).toFixed(1) : value.toFixed(2))
          : value;

        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="glass-card p-4 relative overflow-hidden group hover:border-white/10 transition-colors"
          >
            <div className={`p-2 rounded-lg bg-white/10 w-fit mb-3 ${config.color} shadow-lg`}>
              <config.icon size={18} />
            </div>
            
            <div className="text-[10px] text-white/50 uppercase font-bold tracking-widest leading-none mb-1">
              {config.label}
            </div>
            
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-mono font-bold text-white/90">{displayValue}</span>
              <span className="text-[10px] text-white/40 font-bold">{config.unit}</span>
            </div>

            {/* Subtle graph background decoration */}
            <div className="absolute -bottom-2 -right-2 opacity-[0.03] rotate-12 transition-transform group-hover:scale-110">
              <config.icon size={64} />
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
