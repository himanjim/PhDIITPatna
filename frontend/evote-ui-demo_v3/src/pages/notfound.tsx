/** Simple fallback view shown when no registered route matches the current path. */
export function NotFound() {
  return (
    <div class="card">
      <h2>Not found</h2>
      <a class="badge" href="#/">Back to home</a>
    </div>
  );
}
