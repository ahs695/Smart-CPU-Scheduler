import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { cn } from '../lib/utils';

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
  '#8b5cf6', '#ec4899', '#06b6d4', '#6366f1'
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 6;
const WHEEL_ZOOM_STEP = 0.25;
const BUTTON_ZOOM_STEP = 0.5;

const clampZoom = (z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, +z.toFixed(2)));

/**
 * Props:
 *   trace          per-tick simulator states from /simulate
 *   numCores       number of core rows
 *   onTimeChange   optional callback(currentTime) — drives the SLA alert feed
 *   deadlines      optional { pid: deadline } — shown in block tooltips
 *   missedPids     optional Set of pids that missed their deadline (red outline)
 */
export default function GanttChart({ trace, numCores, onTimeChange, deadlines = {}, missedPids }) {
  const [currentTime, setCurrentTime] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Zoom / pan state, shared by every core track and the time axis
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, panX: 0 });
  const tracksRef = useRef(null);

  const lastIndex = trace.length - 1;
  const atEnd = currentTime >= lastIndex;
  // Derived rather than stored, so reaching the end needs no state update in an effect
  const playing = isPlaying && !atEnd;

  const applyZoom = (updater) => {
    setZoom((prev) => {
      const next = clampZoom(updater(prev));
      if (next === MIN_ZOOM) setPanX(0);
      return next;
    });
  };

  // Wheel-to-zoom over the whole tracks area. Attached natively with
  // { passive: false } because React's onWheel is passive and can't preventDefault.
  useEffect(() => {
    const el = tracksRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? WHEEL_ZOOM_STEP : -WHEEL_ZOOM_STEP;
      setZoom((prev) => {
        const next = clampZoom(prev + delta);
        if (next === MIN_ZOOM) setPanX(0);
        return next;
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Drag-to-pan (only when zoomed in). panX ranges over [-(zoom-1)·trackWidth, 0].
  const trackHandlers = {
    onMouseDown: (e) => {
      if (zoom <= 1) return;
      isPanning.current = true;
      panStart.current = { x: e.clientX, panX };
      e.currentTarget.style.cursor = 'grabbing';
    },
    onMouseMove: (e) => {
      if (!isPanning.current) return;
      const maxPan = e.currentTarget.offsetWidth * (zoom - 1);
      const dx = e.clientX - panStart.current.x;
      setPanX(Math.min(0, Math.max(-maxPan, panStart.current.panX + dx)));
    },
    onMouseUp: (e) => {
      isPanning.current = false;
      e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
    },
    onMouseLeave: (e) => {
      isPanning.current = false;
      e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
    },
  };

  const resetZoom = () => { setZoom(1); setPanX(0); };

  const togglePlay = () => {
    if (playing) {
      setIsPlaying(false);
    } else {
      if (atEnd) setCurrentTime(-1); // replay from the start
      setIsPlaying(true);
    }
  };

  // Playback
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setCurrentTime((prev) => Math.min(prev + 1, lastIndex));
    }, 100 / speed);
    return () => clearInterval(timer);
  }, [playing, lastIndex, speed]);

  // Report playback position to the parent (SLA alerts follow the replay)
  useEffect(() => {
    onTimeChange?.(currentTime);
  }, [currentTime, onTimeChange]);

  const currentState = trace[currentTime] || { cores: [], time: 0, ready_queue: [] };
  const timeScale = Math.max(trace.length, 100);

  const coreSegments = useMemo(() => {
    const segments = Array.from({ length: numCores }, () => []);
    // pid of the segment still open on each core; an idle tick closes it, so
    // the same process resuming after an idle gap starts a new block
    const openPid = Array(numCores).fill(null);
    for (let t = 0; t <= currentTime; t++) {
      const step = trace[t];
      if (!step) continue;
      step.cores.forEach((c) => {
        const coreId = c.id;
        if (coreId >= numCores) return;
        if (c.pid === null) {
          openPid[coreId] = null;
          return;
        }
        const coreSegs = segments[coreId];
        if (openPid[coreId] === c.pid) {
          coreSegs[coreSegs.length - 1].end = step.time;
        } else {
          coreSegs.push({
            pid: c.pid,
            start: step.time,
            end: step.time,
            color: COLORS[c.pid % COLORS.length],
          });
          openPid[coreId] = c.pid;
        }
      });
    }
    return segments;
  }, [trace, currentTime, numCores]);

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <header className="flex items-center justify-between p-4 glass-card border border-mtx1 bg-card/30 backdrop-blur-md rounded-2xl shadow-xl">
        <div className="flex items-center gap-4">
          <button
            onClick={togglePlay}
            className="p-3 rounded-full bg-primary/20 hover:bg-primary/30 text-primary transition-colors"
            title={playing ? 'Pause' : atEnd ? 'Replay' : 'Play'}
          >
            {playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
          </button>
          <button
            onClick={() => { setCurrentTime(-1); setIsPlaying(false); }}
            className="p-3 rounded-full hover:bg-mtx1 transition-colors"
            title="Reset"
          >
            <RotateCcw size={20} />
          </button>
          <div className="flex items-center gap-2 ml-4">
            <span className="text-[10px] text-primary/80 uppercase font-bold tracking-widest">Playback Speed</span>
            {[1, 2, 5, 10].map((s) => (
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
        <div className="text-center">
          <div className="text-[10px] text-secondary/70 uppercase font-bold tracking-widest leading-none mb-1">Ready Queue</div>
          <div className="text-2xl font-mono text-secondary font-bold">{currentState.ready_queue.length}</div>
        </div>
      </header>

      {/* Gantt area — tracks stretch horizontally with zoom */}
      <div className="relative p-6 bg-card/50 border border-ink/5 backdrop-blur-xl rounded-2xl shadow-2xl min-h-[300px]">

        {/* Zoom controls overlay */}
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-field/70 backdrop-blur-sm rounded-xl px-2 py-1 border border-ink/10">
          <button
            onClick={() => applyZoom((z) => z + BUTTON_ZOOM_STEP)}
            className="p-1.5 rounded-lg hover:bg-ink/10 text-mtx3 hover:text-mtx5 transition-colors"
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <span className="text-[10px] font-mono font-bold text-mtx3 w-8 text-center">
            {zoom.toFixed(1)}x
          </span>
          <button
            onClick={() => applyZoom((z) => z - BUTTON_ZOOM_STEP)}
            className="p-1.5 rounded-lg hover:bg-ink/10 text-mtx3 hover:text-mtx5 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
          <div className="w-px h-4 bg-ink/10 mx-1" />
          <button
            onClick={resetZoom}
            className="p-1.5 rounded-lg hover:bg-ink/10 text-mtx3 hover:text-mtx5 transition-colors"
            title="Reset zoom"
          >
            <Maximize2 size={14} />
          </button>
        </div>

        {zoom === 1 && (
          <div className="absolute bottom-3 right-3 z-20 text-[9px] text-mtx2 font-medium italic flex items-center gap-1 pointer-events-none">
            <ZoomIn size={10} /> Scroll to zoom · Drag to pan
          </div>
        )}

        <div ref={tracksRef} className="flex flex-col gap-4">
          {coreSegments.map((segments, coreIdx) => (
            <div key={coreIdx} className="flex items-center gap-4">

              {/* Core label — fixed, never scrolls */}
              <div className="w-24 shrink-0 font-bold text-sm text-mtx3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary" />
                CORE {coreIdx}
              </div>

              {/* Track viewport — fixed visible width, clips the stretched track */}
              <div
                className="relative flex-1 bg-mtx1 rounded-xl h-14 border border-mtx1"
                style={{ overflow: 'hidden', cursor: zoom > 1 ? 'grab' : 'default' }}
                {...trackHandlers}
              >
                {/* Inner track — stretches with zoom, shifts with pan */}
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    bottom: 0,
                    left: `${panX}px`,
                    width: `${zoom * 100}%`,
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
                          left: `${(seg.start / timeScale) * 100}%`,
                          width: `${((seg.end - seg.start + 1) / timeScale) * 100}%`,
                          backgroundColor: seg.color,
                          zIndex: 10,
                          ...(missedPids?.has(seg.pid) && {
                            outline: '2px solid #ef4444',
                            outlineOffset: '-2px',
                          }),
                        }}
                        title={`P${seg.pid}: t=${seg.start}–${seg.end + 1}`
                          + (deadlines[seg.pid] != null ? ` · deadline ${deadlines[seg.pid]}` : '')
                          + (missedPids?.has(seg.pid) ? ' · MISSED' : '')}
                      >
                        <span
                          className="text-[10px] font-bold text-white whitespace-nowrap select-none"
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

          {/* Time axis — stretches and pans with the tracks; ticks line up with the dividers */}
          <div className="flex items-start gap-4">
            <div className="w-24 shrink-0 text-[10px] text-mtx2 font-bold uppercase tracking-widest pt-0.5">
              Time (ticks)
            </div>
            <div className="relative flex-1 overflow-hidden h-5">
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: `${panX}px`,
                  width: `${zoom * 100}%`,
                }}
              >
                {Array.from({ length: 11 }).map((_, i) => (
                  <span
                    key={i}
                    className="absolute text-[10px] text-mtx3 font-mono font-bold tracking-tighter"
                    style={{
                      left: `${i * 10}%`,
                      transform: i === 0 ? 'none' : i === 10 ? 'translateX(-100%)' : 'translateX(-50%)',
                    }}
                  >
                    {Math.round((timeScale / 10) * i)}
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
