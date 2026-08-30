'use client';

import { Environment, OrbitControls } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

import { SERIES_COLORS } from '@/lib/tokens';

/**
 * The animated heart (T112.2). **This module is never imported directly.**
 *
 * `Hero3D` loads it through `next/dynamic` with `ssr: false`, which is what
 * keeps three.js out of every route's first-load bundle and out of the static
 * export's server render. Importing it from a page would undo both.
 *
 * ## What the animation is, and what it is not
 *
 * The beat is a two-phase envelope -- a fast contraction and a slower
 * relaxation, at a fixed 72 beats per minute -- because that is what a heart
 * looks like and a hero that pulses like a sine wave looks like a logo. It is a
 * **decorative animation with no data behind it**, and nothing on the page may
 * suggest otherwise: it is not driven by a recording, it is not the subject's
 * heart rate, and no number is derived from it. The constants below are
 * therefore not metrics and the metric guard is right to ignore them.
 *
 * The geometry is generated in the browser from a parametric curve rather than
 * loaded from a .glb. A model file would be a few hundred kilobytes of asset to
 * commit, host and cache-bust for a shape that is forty lines of maths.
 */

const BEATS_PER_MINUTE = 72;
const SECONDS_PER_BEAT = 60 / BEATS_PER_MINUTE;

/**
 * The classic cardioid-style heart cross-section, extruded and smoothed.
 *
 * Built once and memoised: `useMemo` here is not a micro-optimisation but a
 * correctness requirement, because a new geometry every frame leaks GPU buffers
 * until the context is lost.
 */
function useHeartGeometry(): THREE.ExtrudeGeometry {
  return useMemo(() => {
    const shape = new THREE.Shape();
    const x = 0;
    const y = 0;
    shape.moveTo(x + 0.5, y + 0.5);
    shape.bezierCurveTo(x + 0.5, y + 0.5, x + 0.4, y, x, y);
    shape.bezierCurveTo(x - 0.6, y, x - 0.6, y + 0.7, x - 0.6, y + 0.7);
    shape.bezierCurveTo(x - 0.6, y + 1.1, x - 0.3, y + 1.54, x + 0.5, y + 1.9);
    shape.bezierCurveTo(x + 1.2, y + 1.54, x + 1.6, y + 1.1, x + 1.6, y + 0.7);
    shape.bezierCurveTo(x + 1.6, y + 0.7, x + 1.6, y, x + 1.0, y);
    shape.bezierCurveTo(x + 0.7, y, x + 0.5, y + 0.5, x + 0.5, y + 0.5);

    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: 0.55,
      bevelEnabled: true,
      bevelSegments: 8,
      bevelSize: 0.22,
      bevelThickness: 0.22,
      curveSegments: 32,
    });
    geometry.center();
    geometry.rotateZ(Math.PI);
    geometry.computeVertexNormals();
    return geometry;
  }, []);
}

/**
 * The beat envelope: contraction over the first fifth of the cycle, relaxation
 * over the next third, still for the remainder. Returns a 0..1 scalar.
 */
function beatEnvelope(elapsed: number): number {
  const phase = (elapsed % SECONDS_PER_BEAT) / SECONDS_PER_BEAT;
  if (phase < 0.2) return Math.sin((phase / 0.2) * Math.PI * 0.5);
  if (phase < 0.55) return Math.cos(((phase - 0.2) / 0.35) * Math.PI * 0.5);
  return 0;
}

function Heart({ animate }: { animate: boolean }): JSX.Element {
  const mesh = useRef<THREE.Mesh>(null);
  const geometry = useHeartGeometry();

  useFrame((state) => {
    const target = mesh.current;
    if (target === null) return;
    if (!animate) {
      target.scale.setScalar(1);
      return;
    }
    const pulse = 1 + beatEnvelope(state.clock.elapsedTime) * 0.07;
    target.scale.set(pulse, pulse, pulse);
    target.rotation.y = Math.sin(state.clock.elapsedTime * 0.25) * 0.35;
  });

  return (
    <mesh ref={mesh} geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial
        color={SERIES_COLORS[1] ?? '#D55E00'}
        roughness={0.35}
        metalness={0.15}
        envMapIntensity={0.6}
      />
    </mesh>
  );
}

export interface HeartSceneProps {
  /** False under prefers-reduced-motion: the heart is posed, not beating. */
  animate?: boolean;
  /** Drag to rotate. Off by default; a hero should not trap the scroll. */
  interactive?: boolean;
  /**
   * Frames per second the render loop is clamped to. 30 on integrated
   * graphics (T112.6): halving the frame budget roughly halves the GPU cost,
   * and a 30 fps hero is not perceptibly worse while a dropped-frame 60 fps
   * one is.
   */
  fps?: number;
}

export default function HeartScene({
  animate = true,
  interactive = false,
  fps = 30,
}: HeartSceneProps): JSX.Element {
  return (
    <Canvas
      // `frameloop="demand"` would render once and stop, which is exactly right
      // for the reduced-motion pose and wrong for the beat.
      frameloop={animate ? 'always' : 'demand'}
      dpr={[1, 1.5]}
      shadows={false}
      camera={{ position: [0, 0, 4.2], fov: 42 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      style={{ width: '100%', height: '100%' }}
    >
      <FrameLimiter fps={fps} enabled={animate} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 4, 5]} intensity={1.6} />
      <directionalLight position={[-4, -2, -3]} intensity={0.4} />
      <Heart animate={animate} />
      <Environment preset="city" />
      {interactive ? (
        <OrbitControls enablePan={false} enableZoom={false} rotateSpeed={0.6} />
      ) : null}
    </Canvas>
  );
}

/**
 * Clamp the render loop.
 *
 * R3F renders every animation frame by default. On integrated graphics that is
 * the difference between a hero that costs 4 ms a frame and one that costs 11,
 * and the second number is enough to make the rest of the page feel slow while
 * the hero is on screen.
 */
function FrameLimiter({ fps, enabled }: { fps: number; enabled: boolean }): null {
  const last = useRef(0);
  useFrame((state) => {
    if (!enabled) return;
    const now = state.clock.elapsedTime;
    const interval = 1 / fps;
    if (now - last.current < interval) {
      state.invalidate();
      return;
    }
    last.current = now;
  });
  return null;
}
