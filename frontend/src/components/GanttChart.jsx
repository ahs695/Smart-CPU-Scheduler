import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { cn } from '../lib/utils';

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', 
  '#8b5cf6', '#ec4899', '#06b6d4', '#6366f1'
];

export default function GanttChart({ trace, numCores, totalTime }) {
  const [currentTime, setCurrentTime] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Zoom state
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, panX: 0 });
  const chartWrapperRef = useRef(null);

  const MIN_ZOOM = 1;
  const MAX_ZOOM = 6;

  // Wheel to zoom, scoped to chart area only
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom(prev => {
      const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev + delta));
      // Reset pan if zoomed all the way out
      if (next === MIN_ZOOM) setPanX(0);
      return next;
    });
  }, []);

  useEffect(() => {
    const el = chartWrapperRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Mouse drag to pan (only when zoomed in)
  const handleMouseDown = (e) => {
    if (zoom <= 1) return;
    isPanning.current = true;
    panStart.current = { x: e.clientX, panX };
    e.currentTarget.style.cursor = 'grabbing';
  };

  const handleMouseMove = (e) => {
    if (!isPanning.current) return;
    const dx = e.clientX - panStart.current.x;
    const maxPan = (chartWrapperRef.current?.offsetWidth * (zoom - 1)) / 2 || 0;
    setPanX(Math.min(maxPan, Math.max(-maxPan, panStart.current.panX + dx)));
  };

  const handleMouseUp = (e) => {
    isPanning.current = false;
    e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
  };

  const resetZoom = () => { setZoom(1); setPanX(0); };

  // Playback
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
      {/* Controls — unchanged */}
      <header className="flex items-center justify-between p-4 glass-card border border-mtx1 bg-card/30 backdrop-blur-md rounded-2xl shadow-xl">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-3 rounded-full bg-primary/20 hover:bg-primary/30 text-primary transition-colors"
          >
            {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
          </button>
          <button
            onClick={() => { setCurrentTime(-1); setIsPlaying(false); }}
            className="p-3 rounded-full hover:bg-mtx2 transition-colors"
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
                  speed === s ? "bg-primary text-black shadow-[0_0_10px_rgba(59,130,246,0.5)]" : "hover:bg-mtx1 text-mtx3"
                )}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
          <div className="text-center">
            <div className="text-[10px] text-primary/70 uppercase font-bold tracking-widest leading-none mb-1">Simulation Time</div>
            <div className="text-2xl font-mono text-primary font-bold">
              {currentTime + 1} <span className="text-sm text-mtx2">/ {trace.length}</span>
            </div>
          </div>
          <div className="h-10 bg-mtx1">
          <div className="text-center">
            <div className="text-[10px] text-secondary/70 uppercase font-bold tracking-widest leading-none mb-1">Ready Queue</div>
            <div className="text-2xl font-mono text-secondary font-bold">{currentState.ready_queue.length}</div>
          </div>
        </div>
      </header>

      {/* Gantt Area with zoom */}
      {/* Gantt Area with zoom */}
{/* Gantt Area — horizontal stretch zoom */}
<div className="relative p-6 bg-card/50 border border-white/5 backdrop-blur-xl rounded-2xl shadow-2xl min-h-[300px]">
  
  {/* Zoom controls overlay */}
  <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-black/30 backdrop-blur-sm rounded-xl px-2 py-1 border border-white/10">
    <button
      onClick={() => setZoom(z => Math.min(MAX_ZOOM, +(z + 0.5).toFixed(1)))}
      className="p-1.5 rounded-lg hover:bg-white/10 text-mtx3 hover:text-white transition-colors"
      title="Zoom In"
    >
      <ZoomIn size={14} />
    </button>
    <span className="text-[10px] font-mono font-bold text-mtx3 w-8 text-center">
      {zoom.toFixed(1)}x
    </span>
    <button
      onClick={() => setZoom(z => { const n = Math.max(MIN_ZOOM, +(z - 0.5).toFixed(1)); if (n === 1) setPanX(0); return n; })}
      className="p-1.5 rounded-lg hover:bg-white/10 text-mtx3 hover:text-white transition-colors"
      title="Zoom Out"
    >
      <ZoomOut size={14} />
    </button>
    <div className="w-px h-4 bg-white/10 mx-1" />
    <button
      onClick={resetZoom}
      className="p-1.5 rounded-lg hover:bg-white/10 text-mtx3 hover:text-white transition-colors"
      title="Reset"
    >
      <Maximize2 size={14} />
    </button>
  </div>

  {zoom === 1 && (
    <div className="absolute bottom-3 right-3 z-20 text-[9px] text-mtx2 font-medium italic flex items-center gap-1 pointer-events-none">
      <ZoomIn size={10} /> Scroll to zoom · Drag to pan
    </div>
  )}

  <div className="flex flex-col gap-4">
    {coreSegments.map((segments, coreIdx) => (
      <div key={coreIdx} className="flex items-center gap-4">
        
        {/* Core label — always fixed, never scrolls */}
        <div className="w-24 shrink-0 font-bold text-sm text-mtx3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-primary" />
          CORE {coreIdx}
        </div>

        {/* Track container — fixed visible width, clips overflow */}
        <div
          ref={chartWrapperRef}
          className="relative flex-1 bg-mtx1 rounded-xl h-14 border border-mtx1"
          style={{ overflow: 'hidden' }}
          onWheel={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const delta = e.deltaY < 0 ? 0.25 : -0.25;
            setZoom(prev => {
              const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev + delta));
              if (next === MIN_ZOOM) setPanX(0);
              return next;
            });
          }}
          onMouseDown={(e) => {
            if (zoom <= 1) return;
            isPanning.current = true;
            panStart.current = { x: e.clientX, panX };
            e.currentTarget.style.cursor = 'grabbing';
          }}
          onMouseMove={(e) => {
            if (!isPanning.current) return;
            const trackWidth = e.currentTarget.offsetWidth;
            const maxPan = trackWidth * (zoom - 1);
            const dx = e.clientX - panStart.current.x;
            setPanX(Math.min(0, Math.max(-maxPan, panStart.current.panX + dx)));
          }}
          onMouseUp={(e) => {
            isPanning.current = false;
            e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
          }}
          onMouseLeave={(e) => {
            isPanning.current = false;
            e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
          }}
          style={{
            overflow: 'hidden',
            cursor: zoom > 1 ? 'grab' : 'default',
          }}
        >
          {/* Inner track — this is what stretches horizontally */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${panX}px`,
              width: `${zoom * 100}%`,   // stretches with zoom
            }}
          >
            {/* Time dividers */}
            {Array.from({ length: 10 }).map((_, i) => (
              <div
                key={i}
                className="absolute top-0 bottom-0 border-r border-mtx1"
                style={{ left: `${(i + 1) * 10}%` }}
              />
            ))}

            {/* Process blocks */}
            <AnimatePresence>
              {segments.map((seg) => (
                <motion.div
                  key={`${seg.pid}-${seg.start}`}
                  initial={{ scaleX: 0, opacity: 0 }}
                  animate={{ scaleX: 1, opacity: 1 }}
                  className="absolute top-1 bottom-1 rounded-lg flex items-center justify-center shadow-lg overflow-hidden"
                  style={{
                    left: `${(seg.start / Math.max(trace.length, 100)) * 100}%`,
                    width: `${((seg.end - seg.start + 1) / Math.max(trace.length, 100)) * 100}%`,
                    backgroundColor: seg.color,
                    zIndex: 10,
                  }}
                >
                  <span className="text-[10px] font-bold text-white whitespace-nowrap select-none"
                    style={{ textShadow: '0 1px 3px rgba(0,0,0,0.5)' }}
                  >
                    P{seg.pid}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>
    ))}

    {/* X-Axis — also stretches and pans with zoom */}
    <div className="flex items-center gap-4">
      {/* Spacer matching core label width */}
      <div className="w-24 shrink-0" />

      {/* Timeline container — clips and pans same as tracks */}
      <div className="relative flex-1 overflow-hidden">
        <div
          style={{
            position: 'relative',
            left: `${panX}px`,
            width: `${zoom * 100}%`,
            display: 'flex',
            justifyContent: 'space-between',
            paddingLeft: '4px',
            paddingRight: '4px',
          }}
        >
          {Array.from({ length: 11 }).map((_, i) => (
            <span key={i} className="text-[10px] text-mtx3 font-mono font-bold tracking-tighter">
              {Math.round((Math.max(trace.length, 100) / 10) * i)} ms
            </span>
          ))}
        </div>
      </div>
    </div>

  </div>
</div>
          </div>
        
  );
}