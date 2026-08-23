const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) fail('Usage: node ua-tour-analyze.js <input.json> <output.json>');

  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.nodes || [];
  const edges = input.edges || [];
  const layers = input.layers || [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));

  for (const edge of edges) {
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
  }

  const rank = (counts, key) => nodes
    .map((node) => ({id: node.id, [key]: counts.get(node.id) || 0, name: node.name}))
    .sort((a, b) => b[key] - a[key] || a.id.localeCompare(b.id));
  const allFanIn = rank(fanIn, 'fanIn');
  const allFanOut = rank(fanOut, 'fanOut');
  const topOutCount = Math.max(1, Math.ceil(nodes.length * 0.10));
  const bottomInCount = Math.max(1, Math.ceil(nodes.length * 0.25));
  const topOutIds = new Set(allFanOut.slice(0, topOutCount).map((item) => item.id));
  const bottomInIds = new Set([...allFanIn].reverse().slice(0, bottomInCount).map((item) => item.id));
  const entryNames = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js', 'server.ts', 'server.js',
    'mod.rs', 'main.go', 'main.py', 'main.rs', 'manage.py', 'app.py', 'wsgi.py', 'asgi.py',
    'run.py', '__main__.py', 'Application.java', 'Main.java', 'Program.cs', 'config.ru',
    'index.php', 'App.swift', 'Application.kt', 'main.cpp', 'main.c'
  ]);

  const candidates = [];
  for (const node of nodes) {
    const path = (node.filePath || '').replace(/\\/g, '/');
    const depth = path.split('/').filter(Boolean).length;
    let score = 0;
    if (node.type === 'file') {
      if (entryNames.has(node.name)) score += 3;
      if (depth <= 2) score += 1;
      if (topOutIds.has(node.id)) score += 1;
      if (bottomInIds.has(node.id)) score += 1;
    } else if (node.type === 'document') {
      if (path === 'README.md') score += 5;
      else if (depth === 1 && path.toLowerCase().endsWith('.md')) score += 2;
    }
    if (score > 0) candidates.push({id: node.id, score, name: node.name, summary: node.summary || ''});
  }
  candidates.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));

  const topCode = candidates.find((candidate) => byId.get(candidate.id)?.type === 'file');
  const allowedTraversalEdges = new Set(['imports', 'calls']);
  const adjacency = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) {
    if (allowedTraversalEdges.has(edge.type) && adjacency.has(edge.source) && byId.has(edge.target)) {
      adjacency.get(edge.source).push(edge.target);
    }
  }
  for (const values of adjacency.values()) values.sort();
  const order = [];
  const depthMap = {};
  const byDepth = {};
  if (topCode) {
    const queue = [topCode.id];
    depthMap[topCode.id] = 0;
    while (queue.length) {
      const current = queue.shift();
      const depth = depthMap[current];
      order.push(current);
      if (!byDepth[depth]) byDepth[depth] = [];
      byDepth[depth].push(current);
      for (const next of adjacency.get(current) || []) {
        if (depthMap[next] === undefined) {
          depthMap[next] = depth + 1;
          queue.push(next);
        }
      }
    }
  }

  const projectNode = (node) => ({id: node.id, name: node.name, type: node.type, summary: node.summary || ''});
  const nonCodeFiles = {
    documentation: nodes.filter((node) => node.type === 'document').map(projectNode),
    infrastructure: nodes.filter((node) => ['service', 'pipeline', 'resource'].includes(node.type)).map(projectNode),
    data: nodes.filter((node) => ['table', 'schema', 'endpoint'].includes(node.type)).map(projectNode),
    config: nodes.filter((node) => node.type === 'config').map(projectNode)
  };

  const directional = new Set(edges
    .filter((edge) => ['imports', 'calls'].includes(edge.type))
    .map((edge) => `${edge.source}\u0000${edge.target}\u0000${edge.type}`));
  const pairClusters = [];
  const seenPairs = new Set();
  for (const edge of edges) {
    if (!['imports', 'calls'].includes(edge.type)) continue;
    if (!directional.has(`${edge.target}\u0000${edge.source}\u0000${edge.type}`)) continue;
    const pair = [edge.source, edge.target].sort();
    const key = pair.join('\u0000');
    if (!seenPairs.has(key)) {
      seenPairs.add(key);
      pairClusters.push(new Set(pair));
    }
  }
  const undirectedNeighbors = new Map(nodes.map((node) => [node.id, new Set()]));
  for (const edge of edges) {
    if (!['imports', 'calls'].includes(edge.type)) continue;
    undirectedNeighbors.get(edge.source)?.add(edge.target);
    undirectedNeighbors.get(edge.target)?.add(edge.source);
  }
  for (const cluster of pairClusters) {
    let changed = true;
    while (changed && cluster.size < 5) {
      changed = false;
      for (const node of nodes) {
        if (cluster.has(node.id)) continue;
        const links = [...cluster].filter((id) => undirectedNeighbors.get(node.id)?.has(id)).length;
        if (links >= 2) {
          cluster.add(node.id);
          changed = true;
          if (cluster.size >= 5) break;
        }
      }
    }
  }
  const clusters = pairClusters.map((cluster) => {
    const ids = [...cluster].sort();
    const set = new Set(ids);
    const edgeCount = edges.filter((edge) => set.has(edge.source) && set.has(edge.target)).length;
    return {nodes: ids, edgeCount};
  }).sort((a, b) => b.edgeCount - a.edgeCount || a.nodes.join().localeCompare(b.nodes.join())).slice(0, 10);

  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [node.id, {
    name: node.name,
    type: node.type,
    summary: node.summary || ''
  }]));
  const output = {
    scriptCompleted: true,
    entryPointCandidates: candidates.slice(0, 5),
    fanInRanking: allFanIn.slice(0, 20),
    fanOutRanking: allFanOut.slice(0, 20),
    bfsTraversal: {startNode: topCode?.id || null, order, depthMap, byDepth},
    nonCodeFiles,
    clusters,
    layers: {count: layers.length, list: layers.map(({id, name, description}) => ({id, name, description}))},
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
} catch (error) {
  fail(error.stack || String(error));
}
