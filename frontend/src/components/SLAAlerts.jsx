import React, { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlarmClock, CheckCircle2, ShieldAlert } from 'lucide-react';
import { cn } from '../lib/utils';

/**
 * Live SLA breach feed. A deadline breach becomes certain at the deadline tick,
 * so alerts appear as the Gantt replay passes each missed deadline.
 *
 * Props:
 *   events       deadline_events from /simulate ({pid, program, deadline, completion, lateness, missed, time})
 *   currentTime  Gantt playback position (-1 = not started)
 *   traceLength  number of ticks in the run (to know when playback has finished)
 *   realtime     optional schedulability summary for periodic task sets
 */
export default function SLAAlerts({ events = [], currentTime, traceLength, realtime }) {
  const missed = useMemo(() => events.filter((e) => e.missed), [events]);
  const visible = missed.filter((e) => e.time <= currentTime).reverse();
  const finished = currentTime >= traceLength - 1;
  const tracked = events.length;

  if (!tracked) {
    return (
      <div className="glass-card glass-card--inset-border p-5 text-xs text-mtx3 flex items-center gap-3">
        <AlarmClock size={16} className="text-mtx2" />
        No deadlines in this workload. Add some with <span className="font-semibold text-mtx4">Auto Deadlines</span>,
        a real-time preset, or the Deadline column to see SLA alerts.
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--inset-border p-5 flex flex-col gap-3" data-testid="sla-alerts">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <ShieldAlert size={16} className={visible.length ? 'text-red-400' : 'text-emerald-400'} />
          <h3 className="text-sm font-bold uppercase tracking-widest text-mtx4">SLA Breach Alerts</h3>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span
            className={cn(
              'px-2 py-0.5 rounded-full border font-bold',
              visible.length
                ? 'bg-red-500/15 text-red-400 border-red-500/30'
                : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
            )}
            data-testid="sla-count"
          >
            {visible.length} breach{visible.length === 1 ? '' : 'es'}
          </span>
          <span className="text-mtx3">of {tracked} deadline jobs</span>
        </div>
      </div>

      {realtime && (
        <div className="text-[11px] text-mtx3 leading-relaxed border border-ink/[0.06] rounded-lg px-3 py-2">
          Periodic task set: U = <span className="font-mono text-mtx4">{realtime.utilisation.toFixed(3)}</span>
          {realtime.exact ? '' : ` (${realtime.per_core_utilisation.toFixed(3)} per core)`} · RMS bound{' '}
          <span className="font-mono text-mtx4">{realtime.rms_bound.toFixed(3)}</span> →{' '}
          <span className={realtime.rms_guaranteed ? 'text-emerald-400 font-semibold' : 'text-yellow-400 font-semibold'}>
            RMS {realtime.rms_guaranteed ? 'guaranteed schedulable' : 'not guaranteed'}
          </span>
          {' · '}
          <span className={realtime.edf_feasible ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold'}>
            EDF {realtime.edf_feasible ? 'feasible (U ≤ 1)' : 'overloaded (U > 1)'}
          </span>
          {!realtime.exact && <span className="italic"> — multi-core figures are a heuristic</span>}
        </div>
      )}

      <div className="max-h-[220px] overflow-y-auto flex flex-col gap-1.5 pr-1">
        <AnimatePresence initial={false}>
          {visible.map((e) => (
            <motion.div
              key={e.pid}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center justify-between gap-3 text-xs bg-red-500/[0.07] border border-red-500/20 rounded-lg px-3 py-2"
            >
              <span className="text-red-300">
                <span className="font-mono font-bold">P{e.pid}</span>
                {e.program && <span className="text-mtx3"> ({e.program})</span>} missed its deadline at t={e.deadline}
              </span>
              <span className="font-mono text-red-400 whitespace-nowrap">
                {e.completion != null ? `+${e.lateness} late` : 'never finished'}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
        {currentTime < 0 && (
          <p className="text-[11px] text-mtx2 italic">Press play on the Gantt chart — breaches appear as their deadlines pass.</p>
        )}
        {finished && missed.length === 0 && (
          <p className="text-xs text-emerald-400 flex items-center gap-2">
            <CheckCircle2 size={14} /> Every deadline met.
          </p>
        )}
        {finished && missed.length > 0 && (
          <p className="text-[11px] text-mtx3">
            Run finished: {missed.length} of {tracked} jobs missed ({((missed.length / tracked) * 100).toFixed(1)}%),
            worst lateness {Math.max(...missed.map((e) => e.lateness ?? 0))} ticks.
          </p>
        )}
      </div>
    </div>
  );
}
