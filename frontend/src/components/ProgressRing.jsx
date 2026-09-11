// Determinate circular progress: a conic-gradient arc filled to `progress`
// (0-100), with the center punched out to read as a ring rather than a pie.
export default function ProgressRing({ progress = 0, size = 14 }) {
  const pct = Math.max(0, Math.min(100, progress));
  return (
    <span
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        flexShrink: 0,
        display: "inline-block",
        position: "relative",
        background: `conic-gradient(var(--text-info) ${pct * 3.6}deg, var(--s-badge-info-subdued) 0deg)`,
      }}
    >
      <span
        style={{
          position: "absolute",
          inset: 3,
          borderRadius: "50%",
          background: "var(--s-box-base)",
        }}
      />
    </span>
  );
}
