import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from .workflow import run

def main():
    parser = argparse.ArgumentParser(description='Research a topic using arXiv abstracts and multiple agents.')
    parser.add_argument('topic')
    parser.add_argument('--output', type=Path, default=Path('reports/report.md'))
    args = parser.parse_args()
    load_dotenv()
    result = asyncio.run(run(args.topic))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    references = '\n'.join(f"- [{s['id']}] {s['title']} ({s['published'][:10]}): {s['url']}" for s in result['sources'])
    args.output.write_text(result['report'] + '\n\n## Retrieved source records\n\n' + references + '\n', encoding='utf-8')
    print(f'Report written to {args.output}')

if __name__ == '__main__': main()
