// Recreates the two Button2 variants the design actually uses (primary + tertiary,
// tone=neutral), styles taken from the design kit's Button2.jsx.
export default function Button({ variant = "tertiary", label, onClick, type = "button", style }) {
  const isPrimary = variant === "primary";
  return (
    <button
      type={type}
      onClick={onClick}
      style={{
        width: "fit-content",
        border: "none",
        borderRadius: "var(--radius-button)",
        cursor: "pointer",
        fontFamily: "var(--font-sans)",
        fontSize: 12,
        lineHeight: "16px",
        fontWeight: isPrimary ? 600 : 500,
        padding: "6px 12px",
        color: isPrimary ? "var(--s-button-text-on-primary)" : "var(--text-default)",
        background: isPrimary
          ? "linear-gradient(180deg, rgba(48,48,48,0) 63.53%, rgba(255,255,255,0.15) 100%), var(--s-button-primary)"
          : "var(--s-box-base)",
        boxShadow: isPrimary ? "var(--shadow-button-primary-inset)" : "var(--shadow-button-inset)",
        ...style,
      }}
    >
      {label}
    </button>
  );
}
