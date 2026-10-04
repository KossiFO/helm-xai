export default function ProfileSelect({ profiles, value, onChange }) {
  return (
    <div className="profile-grid" role="group" aria-label="Profil de lecture">
      {profiles.map((p) => (
        <button key={p.id} className={`profile-card ${value === p.id ? "selected" : ""}`}
          aria-pressed={value === p.id} onClick={() => onChange(p.id)}>
          <strong>{p.label}</strong><span>{p.description}</span>
        </button>
      ))}
    </div>
  );
}
