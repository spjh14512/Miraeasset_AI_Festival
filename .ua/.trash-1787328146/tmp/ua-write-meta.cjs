const fs = require("fs");

const [outputPath, gitCommitHash, analyzedFiles] = process.argv.slice(2);
const meta = {
  lastAnalyzedAt: new Date().toISOString(),
  gitCommitHash,
  version: "1.0.0",
  analyzedFiles: Number(analyzedFiles),
};
fs.writeFileSync(outputPath, JSON.stringify(meta, null, 2));
