// Recreates the design kit's Avatar.jsx icon-size variant.
export default function Avatar({ initials = "AG", size = 28 }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "var(--radius-button)",
        background: "var(--s-avatar-bg)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontWeight: 500,
          fontSize: 9,
          lineHeight: size + "px",
          color: "var(--s-avatar-text)",
        }}
      >
        {initials}
      </span>
    </div>
  );
}
