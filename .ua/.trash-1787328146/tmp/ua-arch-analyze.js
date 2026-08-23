const fs = require('fs');
const path = require('path');

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

const normalize = (p) => String(p || '').replace(/\\/g, '/').replace(/^\.\//, '');

function commonDirectoryPrefix(paths) {
  const dirs = paths.map((p) => normalize(p).split('/').slice(0, -1));
  if (!dirs.length) return [];
  const prefix = [];
  const limit = Math.min(...dirs.map((parts) => parts.length));
  for (let i = 0; i < limit; i += 1) {
    const value = dirs[0][i];
    if (dirs.every((parts) => parts[i] === value)) prefix.push(value);
    else break;
  }
  return prefix;
}

function flatPattern(filePath) {
  const name = path.posix.basename(normalize(filePath)).toLowerCase();
  if (/^(test_.*\.py|.*\.(test|spec)\.|.*_test\.go|.*test\.java|.*_spec\.rb|.*test\.php|.*tests\.cs)/i.test(name)) return 'test';
  if (/config|settings|pyproject\.toml|package\.json|requirements\.txt/.test(name)) return 'config';
  return (path.posix.extname(name).slice(1) || 'root').toLowerCase();
}

function directoryGroup(filePath, prefix, flat) {
  const parts = normalize(filePath).split('/');
  if (flat) return flatPattern(filePath);
  const rest = parts.slice(prefix.length);
  return rest.length > 1 ? rest[0] : 'root';
}

const patterns = [
  [/^(routes|api|controllers|endpoints|handlers|serializers|routers|blueprints)$/i, 'api'],
  [/^(services|core|lib|domain|logic|internal|composables|mailers|jobs|channels|signals)$/i, 'service'],
  [/^(models|db|data|persistence|repository|entities|entity|migrations|sql|database|schema)$/i, 'data'],
  [/^(components|views|pages|ui|layouts|screens)$/i, 'ui'],
  [/^(middleware|plugins|interceptors|guards)$/i, 'middleware'],
  [/^(utils|helpers|common|shared|tools|pkg|templatetags)$/i, 'utility'],
  [/^(config|constants|env|settings|management|commands)$/i, 'config'],
  [/^(__tests__|test|tests|spec|specs|src\/test\/java)$/i, 'test'],
  [/^(types|interfaces|schemas|contracts|dtos|dto|request|response)$/i, 'types'],
  [/^(hooks)$/i, 'hooks'],
  [/^(store|state|reducers|actions|slices)$/i, 'state'],
  [/^(assets|static|public)$/i, 'assets'],
  [/^(cmd|bin)$/i, 'entry'],
  [/^(docs|documentation|wiki)$/i, 'documentation'],
  [/^(deploy|deployment|infra|infrastructure|k8s|kubernetes|helm|charts|terraform|tf|docker)$/i, 'infrastructure'],
  [/^(\.github|\.gitlab|\.circleci)$/i, 'ci-cd'],
];

function patternForGroup(group) {
  for (const [regex, label] of patterns) if (regex.test(group)) return label;
  return null;
}

function classifyFile(filePath) {
  const p = normalize(filePath);
  const name = path.posix.basename(p);
  const lower = name.toLowerCase();
  if (/^(test_.*\.py|.*\.(test|spec)\.|.*_test\.go|.*test\.java|.*_spec\.rb|.*test\.php|.*tests\.cs)$/i.test(name)) return 'test';
  if (/\.d\.ts$/i.test(name) || /\.(graphql|gql|proto)$/i.test(name)) return 'types';
  if ((/^index\.(ts|js)$/i.test(name) || name === '__init__.py') || lower === 'manage.py' || /^(config\.ru)$/i.test(name)) return 'entry';
  if (/^(wsgi|asgi)\.py$/i.test(name)) return 'config';
  if (/^(cargo\.toml|go\.mod|gemfile|pom\.xml|build\.gradle|composer\.json|pyproject\.toml)$/i.test(name)) return 'config';
  if (/^dockerfile/i.test(name) || /^docker-compose\./i.test(name) || /\.tf(vars)?$/i.test(name) || lower === 'makefile') return 'infrastructure';
  if (/^\.github\/workflows\//i.test(p) || lower === '.gitlab-ci.yml' || lower === 'jenkinsfile') return 'ci-cd';
  if (/\.sql$/i.test(name)) return 'data';
  if (/\.(md|rst)$/i.test(name)) return 'documentation';
  return null;
}

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.fileNodes || [];
  const imports = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const idToNode = new Map(nodes.map((node) => [node.id, node]));
  const paths = nodes.map((node) => normalize(node.filePath || node.name));
  const prefix = commonDirectoryPrefix(paths);
  const flat = paths.every((p) => !p.includes('/'));
  const idToGroup = new Map();
  const directoryGroups = {};
  for (const node of nodes) {
    const group = directoryGroup(node.filePath || node.name, prefix, flat);
    idToGroup.set(node.id, group);
    (directoryGroups[group] ||= []).push(node.id);
  }

  const nodeTypeGroups = {};
  for (const node of nodes) (nodeTypeGroups[node.type] ||= []).push(node.id);

  const fileFanIn = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const adjacency = Object.fromEntries(nodes.map((node) => [node.id, []]));
  const inter = new Map();
  const incident = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, 0]));
  const internal = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, 0]));
  const groupImports = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, new Set()]));
  const groupImportedBy = Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, new Set()]));
  for (const edge of imports) {
    if (!idToNode.has(edge.source) || !idToNode.has(edge.target)) continue;
    fileFanOut[edge.source] += 1;
    fileFanIn[edge.target] += 1;
    adjacency[edge.source].push(edge.target);
    const from = idToGroup.get(edge.source);
    const to = idToGroup.get(edge.target);
    incident[from] += 1;
    if (to !== from) incident[to] += 1;
    else internal[from] += 1;
    if (from !== to) {
      const key = `${from}\u0000${to}`;
      inter.set(key, (inter.get(key) || 0) + 1);
      groupImports[from].add(to);
      groupImportedBy[to].add(from);
    }
  }
  const interGroupImports = [...inter].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return { from, to, count };
  }).sort((a, b) => b.count - a.count || a.from.localeCompare(b.from));

  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    intraGroupDensity[group] = {
      internalEdges: internal[group],
      totalEdges: incident[group],
      density: incident[group] ? internal[group] / incident[group] : 0,
      importsFromGroups: [...groupImports[group]].sort(),
      importedByGroups: [...groupImportedBy[group]].sort(),
    };
  }

  const crossCounts = new Map();
  const nonCodeConnections = [];
  for (const edge of allEdges) {
    const source = idToNode.get(edge.source);
    const target = idToNode.get(edge.target);
    if (!source || !target) continue;
    const key = `${source.type}\u0000${target.type}\u0000${edge.type}`;
    crossCounts.set(key, (crossCounts.get(key) || 0) + 1);
    if (source.type !== 'file' || target.type !== 'file') nonCodeConnections.push(edge);
  }
  const crossCategoryEdges = [...crossCounts].map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('\u0000');
    return { fromType, toType, edgeType, count };
  }).sort((a, b) => b.count - a.count);

  const reverseInter = new Map(interGroupImports.map((x) => [`${x.from}\u0000${x.to}`, x.count]));
  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const item of interGroupImports) {
    const pair = [item.from, item.to].sort().join('\u0000');
    if (seenPairs.has(pair)) continue;
    seenPairs.add(pair);
    const forward = reverseInter.get(`${item.from}\u0000${item.to}`) || 0;
    const backward = reverseInter.get(`${item.to}\u0000${item.from}`) || 0;
    if (forward > backward) dependencyDirection.push({ dependent: item.from, dependsOn: item.to, count: forward, reverseCount: backward });
    else if (backward > forward) dependencyDirection.push({ dependent: item.to, dependsOn: item.from, count: backward, reverseCount: forward });
    else dependencyDirection.push({ dependent: item.from, dependsOn: item.to, count: forward, reverseCount: backward, bidirectional: true });
  }

  const byClass = (label) => nodes.filter((n) => classifyFile(n.filePath || n.name) === label).map((n) => normalize(n.filePath || n.name));
  const infraFiles = nodes.filter((n) => ['infrastructure', 'ci-cd'].includes(classifyFile(n.filePath || n.name))).map((n) => normalize(n.filePath || n.name));
  const deploymentTopology = {
    hasDockerfile: paths.some((p) => /(^|\/)Dockerfile/i.test(p)),
    hasCompose: paths.some((p) => /(^|\/)docker-compose\./i.test(p)),
    hasK8s: paths.some((p) => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/i.test(p)),
    hasTerraform: paths.some((p) => /\.tf(vars)?$/i.test(p) || /(^|\/)(terraform|tf)(\/|$)/i.test(p)),
    hasCI: paths.some((p) => /(^|\/)(\.github\/workflows|\.gitlab|\.circleci)(\/|$)|(^|\/)(Jenkinsfile|\.gitlab-ci\.yml)$/i.test(p)),
    infraFiles,
  };

  const dataPipeline = {
    schemaFiles: nodes.filter((n) => n.type === 'schema' || /schema|\.(graphql|gql|proto|sql)$/i.test(n.filePath || n.name)).map((n) => normalize(n.filePath || n.name)),
    migrationFiles: paths.filter((p) => /(^|\/)migrations?(\/|$)|migration/i.test(p)),
    dataModelFiles: nodes.filter((n) => /(^|\/)(models?|entities|schemas?)(\/|$)/i.test(n.filePath || n.name) || (n.tags || []).some((t) => /data-model|entity|schema/i.test(t))).map((n) => normalize(n.filePath || n.name)),
    apiHandlerFiles: nodes.filter((n) => n.type === 'endpoint' || /(^|\/)(routes?|api|controllers?|handlers?|endpoints?)(\/|$)/i.test(n.filePath || n.name) || (n.tags || []).some((t) => /api-handler|endpoint|controller/i.test(t))).map((n) => normalize(n.filePath || n.name)),
  };

  const docNodes = nodes.filter((n) => n.type === 'document' || /\.(md|rst)$/i.test(n.filePath || n.name));
  const documented = new Set();
  for (const doc of docNodes) {
    const p = normalize(doc.filePath || doc.name);
    const group = idToGroup.get(doc.id);
    documented.add(group);
    const text = `${doc.summary || ''} ${(doc.tags || []).join(' ')}`.toLowerCase();
    for (const candidate of Object.keys(directoryGroups)) if (text.includes(candidate.toLowerCase())) documented.add(candidate);
    if (/^readme\.md$/i.test(p)) documented.add('root');
  }
  const groups = Object.keys(directoryGroups);
  const docCoverage = {
    groupsWithDocs: groups.filter((g) => documented.has(g)).length,
    totalGroups: groups.length,
    coverageRatio: groups.length ? groups.filter((g) => documented.has(g)).length / groups.length : 0,
    undocumentedGroups: groups.filter((g) => !documented.has(g)),
  };

  const result = {
    scriptCompleted: true,
    commonPathPrefix: prefix.join('/'),
    directoryGroups,
    nodeTypeGroups,
    importAdjacency: adjacency,
    crossCategoryEdges,
    nonCodeConnections,
    interGroupImports,
    intraGroupDensity,
    patternMatches: Object.fromEntries(Object.keys(directoryGroups).map((g) => [g, patternForGroup(g)]).filter(([, v]) => v)),
    filePatternMatches: Object.fromEntries(nodes.map((n) => [n.id, classifyFile(n.filePath || n.name)]).filter(([, v]) => v)),
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: nodes.length,
      filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g, ids]) => [g, ids.length])),
      nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, ids]) => [t, ids.length])),
    },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error.stack || String(error));
  process.exit(1);
}
