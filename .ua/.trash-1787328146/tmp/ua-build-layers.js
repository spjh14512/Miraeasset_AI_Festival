const fs = require('fs');

const [resultsPath, outputPath] = process.argv.slice(2);
if (!resultsPath || !outputPath) {
  console.error('usage: node ua-build-layers.js <arch-results.json> <layers.json>');
  process.exit(1);
}

try {
  const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
  const groups = results.directoryGroups;
  const take = (group, predicate = () => true) => groups[group].filter(predicate);
  const notDocument = (id) => !id.startsWith('document:');
  const is = (expected) => (id) => id === expected;
  const not = (predicate) => (id) => !predicate(id);

  const layers = [
    {
      id: 'layer:agent-workflow',
      name: 'Agent Workflow 레이어',
      description: 'LangGraph 기반 agent graph와 공유 AgentState를 정의하고 workflow 실행 상태를 조율합니다.',
      nodeIds: [...groups.agent_graph],
    },
    {
      id: 'layer:conversion-pipeline',
      name: '문서 변환 파이프라인',
      description: 'DART 원문을 파싱·정규화하여 Canonical Section과 Evidence Fragment로 조립하고 검증하는 핵심 변환 로직입니다.',
      nodeIds: take('converters', notDocument),
    },
    {
      id: 'layer:vector-retrieval',
      name: 'Vector Retrieval 레이어',
      description: 'Evidence Fragment의 contextual text와 embedding point를 구성하여 Qdrant 적재 및 Hybrid Retrieval 입력을 준비합니다.',
      nodeIds: [...groups.vector_db],
    },
    {
      id: 'layer:knowledge-graph',
      name: 'Knowledge Graph 레이어',
      description: 'Canonical Section과 Evidence Fragment를 Neo4j의 Disclosure-Section-Evidence 구조로 변환하고 schema에 맞춰 적재합니다.',
      nodeIds: [...groups.knowledge_graph],
    },
    {
      id: 'layer:pipeline-cli',
      name: 'Pipeline CLI 레이어',
      description: 'Canonical Section 생성, Evidence Fragment 생성·검증, golden 승인 작업을 실행하는 운영 entry-point를 제공합니다.',
      nodeIds: [...groups.scripts, ...take('root', is('file:main.py'))],
    },
    {
      id: 'layer:regression-validation',
      name: '회귀 검증 레이어',
      description: 'converter·vector·storage 동작을 pytest와 golden corpus로 검증하고 후보 snapshot 및 원문 보정 fixture를 관리합니다.',
      nodeIds: [...groups.tests, ...groups.tmp],
    },
    {
      id: 'layer:documentation',
      name: '문서 레이어',
      description: '프로젝트 사용법, converter 구조, 데이터 필터링 기준, ontology 자료와 검증 skill 운용 지침을 설명합니다.',
      nodeIds: [
        ...groups.DOCS,
        ...take('root', is('document:README.md')),
        ...take('converters', is('document:converters/README.md')),
        ...take('.agents', (id) => id.startsWith('document:')),
      ],
    },
    {
      id: 'layer:project-configuration',
      name: '프로젝트 설정',
      description: 'Python runtime과 dependency, 환경 변수 예시, Codex 분석 설정 및 converter 검증 agent 구성을 관리합니다.',
      nodeIds: [
        ...take('root', not((id) => id === 'file:main.py' || id === 'document:README.md')),
        ...take('.agents', not((id) => id.startsWith('document:'))),
        ...groups['.ua'],
      ],
    },
  ];

  const expected = Object.values(groups).flat();
  const assigned = layers.flatMap((layer) => layer.nodeIds);
  const counts = new Map();
  for (const id of assigned) counts.set(id, (counts.get(id) || 0) + 1);
  const missing = expected.filter((id) => !counts.has(id));
  const duplicates = [...counts].filter(([, count]) => count !== 1);
  const invented = assigned.filter((id) => !expected.includes(id));
  if (layers.length < 3 || layers.length > 10) throw new Error(`invalid layer count: ${layers.length}`);
  if (layers.some((layer) => layer.nodeIds.length === 0)) throw new Error('empty layer detected');
  if (assigned.length !== results.fileStats.totalFileNodes || missing.length || duplicates.length || invented.length) {
    throw new Error(JSON.stringify({ expected: expected.length, assigned: assigned.length, missing, duplicates, invented }, null, 2));
  }
  fs.writeFileSync(outputPath, JSON.stringify(layers, null, 2));
  console.log(JSON.stringify(layers.map((layer) => ({ name: layer.name, count: layer.nodeIds.length }))));
} catch (error) {
  console.error(error.stack || String(error));
  process.exit(1);
}
