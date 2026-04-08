import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, FastForward, Info } from 'lucide-react';
import { cn } from '../lib/utils';

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', 
  '#8b5cf6', '#ec4899', '#06b6d4', '#6366f1'
];

export default function GanttChart({ trace, numCores, totalTime }) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  
  // Playback control
  useEffect(() => {
    let timer;
    if (isPlaying && currentTime < trace.length - 1) {
      timer = setInterval(() => {
        setCurrentTime(prev => Math.min(prev + 1, trace.length - 1));
      }, 100 / speed);
    } else if (currentTime >= trace.length - 1) {
      setIsPlaying(false);
    }
    return () => clearInterval(timer);
  }, [isPlaying, currentTime, trace.length, speed]);

  const currentState = trace[currentTime] || { cores: [], time: 0, ready_queue: [] };

  // Calculate blocks for the Gantt Visualization
  // We need to transform the flat trace into a list of "segments" for each core
  const coreSegments = useMemo(() => {
    const segments = Array.from({ length: numCores }, () => []);
    
    for (let t = 0; t <= currentTime; t++) {
      const step = trace[t];
      if (!step) continue;
      
      step.cores.forEach(c => {
        const coreId = c.id;
        if (coreId >= numCores) return;
        
        const lastSegment = segments[coreId][segments[coreId].length - 1];
        if (lastSegment && lastSegment.pid === c.pid) {
          lastSegment.end = step.time;
        } else if (c.pid !== null) {
          segments[coreId].push({
            pid: c.pid,
            start: step.time,
            end: step.time,
            color: COLORS[c.pid % COLORS.length]
          });
        }
      });
    }
    return segments;
  }, [trace, currentTime, numCores]);

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <header className="flex items-center justify-between p-4 glass-card border border-white/5 bg-card/30 backdrop-blur-md rounded-2xl shadow-xl">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-3 rounded-full bg-primary/20 hover:bg-primary/30 text-primary transition-colors"
          >
            {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
          </button>
          
          <button 
            onClick={() => { setCurrentTime(0); setIsPlaying(false); }}
            className="p-3 rounded-full hover:bg-white/5 transition-colors"
          >
            <RotateCcw size={20} />
          </button>

          <div className="flex items-center gap-2 ml-4">
            <span className="text-[10px] text-primary/80 uppercase font-bold tracking-widest">Playback Speed</span>
            {[1, 2, 5, 10].map(s => (
              <button 
                key={s}
                onClick={() => setSpeed(s)}
                className={cn(
                  "px-3 py-1 rounded-md text-sm transition-all font-bold",
                  speed === s ? "bg-primary text-white shadow-[0_0_10px_rgba(59,130,246,0.5)]" : "hover:bg-white/10 text-white/60"
                )}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-[10px] text-primary/70 uppercase font-bold tracking-widest leading-none mb-1">Simulation Time</div>
            <div className="text-2xl font-mono text-primary font-bold">{currentTime} <span className="text-sm text-white/30">/ {trace.length - 1}</span></div>
          </div>
          
          <div className="h-10 w-px bg-white/10" />
          
          <div className="text-right">
            <div className="text-[10px] text-secondary/70 uppercase font-bold tracking-widest leading-none mb-1">Ready Queue</div>
            <div className="text-2xl font-mono text-secondary font-bold">{currentState.ready_queue.length}</div>
          </div>
        </div>
      </header>

      {/* Gantt Area */}
      <div className="p-6 glass-card overflow-x-auto min-h-[300px]">
        <div className="min-w-[800px] flex flex-col gap-4">
          {coreSegments.map((segments, coreIdx) => (
            <div key={coreIdx} className="flex items-center gap-4">
              <div className="w-24 shrink-0 font-bold text-sm text-white/60 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary" />
                CORE {coreIdx}
              </div>
              
              <div className="relative flex-1 bg-white/5 rounded-xl h-14 border border-white/5 overflow-hidden">
                {/* Time Dividers */}
                {Array.from({ length: 10 }).map((_, i) => (
                  <div 
                    key={i} 
                    className="absolute top-0 bottom-0 border-r border-white/5" 
                    style={{ left: `${(i + 1) * 10}%` }}
                  />
                ))}

                {/* Blocks */}
                <AnimatePresence>
                  {segments.map((seg, idx) => (
                    <motion.div
                      key={`${seg.pid}-${seg.start}`}
                      initial={{ scaleX: 0, opacity: 0 }}
                      animate={{ scaleX: 1, opacity: 1 }}
                      className="absolute top-1 bottom-1 rounded-lg flex items-center justify-center text-[10px] font-bold shadow-lg"
                      style={{
                        left: `${(seg.start / Math.max(trace.length, 100)) * 100}%`,
                        width: `${((seg.end - seg.start + 1) / Math.max(trace.length, 100)) * 100}%`,
                        backgroundColor: seg.color,
                        zIndex: 10
                      }}
                    >
                      P{seg.pid}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          ))}

          {/* X-Axis labels */}
          <div className="flex ml-28 mt-4 justify-between px-4 text-[10px] text-white/50 font-mono font-bold tracking-tighter">
            {Array.from({ length: 11 }).map((_, i) => (
              <span key={i}>{Math.round((Math.max(trace.length, 100) / 10) * i)} ms</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
