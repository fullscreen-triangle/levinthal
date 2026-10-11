import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { adjacency, layerById, nodeById, phaseById, statusById } from '../lib/spec.js';
import { dedicatedPages } from './components/index.js';
import GenericComponentPage from './GenericComponentPage.jsx';
import NotFound from './NotFound.jsx';

export default function ComponentRoute() {
  const { id } = useParams();
  const node = nodeById.get(id);

  useEffect(() => {
    document.title = node ? `${node.label} — dcat-ap-plus-p450` : 'Unknown component — dcat-ap-plus-p450';
  }, [node]);

  if (!node) {
    return <NotFound message={`There is no component with id “${id}” in components.json.`} />;
  }

  const Page = dedicatedPages[node.id] ?? GenericComponentPage;
  const { in: incoming, out: outgoing } = adjacency.get(node.id);
  return (
    <Page
      key={node.id}
      node={node}
      layer={layerById.get(node.layer)}
      phase={phaseById.get(node.phase)}
      status={statusById.get(node.status)}
      incoming={incoming}
      outgoing={outgoing}
    />
  );
}
