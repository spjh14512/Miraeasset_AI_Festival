const fs = require('fs');

const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) {
  console.error('usage: node ua-prepare-arch-input.js <assembled-graph.json> <output.json>');
  process.exit(1);
}

try {
  const graph = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
  const allowedTypes = new Set([
    'file', 'config', 'document', 'service', 'pipeline',
    'table', 'schema', 'resource', 'endpoint'
  ]);
  const fileNodes = graph.nodes.filter((node) => allowedTypes.has(node.type));
  const ids = new Set(fileNodes.map((node) => node.id));
  const allEdges = graph.edges.filter(
    (edge) => ids.has(edge.source) && ids.has(edge.target)
  );
  const importEdges = allEdges.filter((edge) => edge.type === 'imports');
  fs.writeFileSync(outputPath, JSON.stringify({ fileNodes, importEdges, allEdges }, null, 2));
} catch (error) {
  console.error(error.stack || String(error));
  process.exit(1);
}
