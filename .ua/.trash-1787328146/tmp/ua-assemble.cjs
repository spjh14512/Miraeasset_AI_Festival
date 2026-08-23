const fs = require("fs");

const [scanPath, graphPath, layersPath, tourPath, commitHash] = process.argv.slice(2);
const scan = JSON.parse(fs.readFileSync(scanPath, "utf8"));
const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
const layers = JSON.parse(fs.readFileSync(layersPath, "utf8"));
const tour = JSON.parse(fs.readFileSync(tourPath, "utf8"));

const complete = {
  version: "1.0.0",
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt: new Date().toISOString(),
    gitCommitHash: commitHash,
  },
  nodes: graph.nodes,
  edges: graph.edges,
  layers,
  tour,
};

fs.writeFileSync(graphPath, JSON.stringify(complete, null, 2));

