// Vercel deploys only this static interface. Python and Ollama stay on the user's machine.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
process.chdir(fileURLToPath(new URL('../', import.meta.url)));
mkdirSync('dist', { recursive: true });
const html = readFileSync('research_assistant/index.html', 'utf8').replace('<head>', '<head><meta name="research-hosting" content="static">');
writeFileSync('dist/index.html', html);
writeFileSync('dist/robots.txt', 'User-agent: *\nAllow: /\n');
console.log('Built static Research Studio; no cloud functions or paid services.');
