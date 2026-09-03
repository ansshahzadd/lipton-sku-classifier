// Recreates the design kit's Badge.jsx tones (base/info/success/critical).
const TONES = {
  neutral: { bg: "var(--s-badge-base)", text: "var(--text-subdued)" },
  info: { bg: "var(--s-badge-info-subdued)", text: "var(--text-info)" },
  success: { bg: "var(--s-badge-success-subdued)", text: "var(--text-success)" },
  critical: { bg: "var(--s-badge-critical-subdued)", text: "var(--text-critical)" },
};

export default function Badge({ tone = "neutral", content }) {
  const t = TONES[tone] || TONES.neutral;
  return (
    <span
      style={{
        width: "fit-content",
        borderRadius: "var(--radius-button)",
        background: t.bg,
        display: "inline-flex",
        padding: "2px 8px",
        fontFamily: "var(--font-sans)",
        fontSize: 12,
        lineHeight: "16px",
        fontWeight: 500,
        whiteSpace: "nowrap",
        color: t.text,
      }}
    >
      {content}
    </span>
  );
}
