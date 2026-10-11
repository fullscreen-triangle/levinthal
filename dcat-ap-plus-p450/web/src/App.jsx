import { useEffect } from 'react';
import { Link, Route, Routes, useLocation } from 'react-router-dom';
import Home from './pages/Home.jsx';
import ComponentRoute from './pages/ComponentRoute.jsx';
import NotFound from './pages/NotFound.jsx';
import { spec, nodes, edges } from './lib/spec.js';

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

export default function App() {
  return (
    <>
      <ScrollToTop />
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-bar">
        <Link to="/" className="site-name">
          dcat-ap-plus-p450
        </Link>
        <span className="site-bar-meta mono">
          components.json v{spec.version} · {spec.updated}
        </span>
      </header>
      <main id="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/component/:id" element={<ComponentRoute />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <footer className="site-footer">
        <p>
          Rendered from <span className="mono">docs/specifications/components.json</span> (v{spec.version},{' '}
          {spec.updated}): {nodes.length} components, {edges.length} connections. Every count on this site is computed
          from that file when the page loads; none is written by hand.
        </p>
        <p>Hosted in Buhera OS. The specification is documented in components-graph.md.</p>
      </footer>
    </>
  );
}
