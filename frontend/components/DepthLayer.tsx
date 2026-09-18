/**
 * The page's background depth (T139.2): one fixed sheet behind everything --
 * a radial gradient mesh, two soft accent orbs, and the existing telemetry
 * grid at low opacity. Rendered once from the root layout, not per page.
 *
 * Three separate layered elements, not one with combined classes: `.depth-mesh`
 * and `.telemetry-grid` both set `background-image`, and stacking them on one
 * element would let whichever class the cascade orders last simply overwrite
 * the other's gradient.
 *
 * A server component with no interactivity and nothing that moves, so it
 * needs no reduced-motion guard: it is atmosphere, not an animation.
 */
export function DepthLayer() {
  return (
    <div aria-hidden="true" className="fixed inset-0 -z-10 overflow-hidden print:hidden">
      <div className="depth-mesh absolute inset-0" />
      <div className="telemetry-grid depth-grid absolute inset-0" />
      <span className="depth-orb -left-[10%] -top-[12%] h-[38vmax] w-[38vmax] bg-accent/14" />
      <span className="depth-orb -bottom-[16%] -right-[8%] h-[42vmax] w-[42vmax] bg-accent-line/16" />
    </div>
  );
}
