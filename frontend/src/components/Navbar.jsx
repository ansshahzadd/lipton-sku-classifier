import { NavLink } from "react-router-dom";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/uploads", label: "Uploads" },
  { to: "/rejected", label: "Rejected" },
  { to: "/dashboard", label: "Dashboard" },
];

export default function Navbar({ brandName }) {
  return (
    <div
      style={{
        position: "sticky",
        top: 0,
        zIndex: 20,
        background: "var(--s-box-base)",
        borderBottom: "1px solid var(--s-border-subdued)",
        height: 56,
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "0 24px",
      }}
    >
      <div style={{ fontSize: 14, fontWeight: 650, letterSpacing: "-0.2px", whiteSpace: "nowrap" }}>{brandName}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
            style={({ isActive }) => ({
              border: "none",
              cursor: "pointer",
              fontFamily: "inherit",
              fontSize: 13,
              lineHeight: "20px",
              padding: "6px 10px",
              borderRadius: 8,
              fontWeight: isActive ? 600 : 550,
              color: isActive ? "var(--text-default)" : "var(--text-subdued)",
            })}
          >
            {item.label}
          </NavLink>
        ))}
      </div>
      <div style={{ flexGrow: 1 }} />
    </div>
  );
}
