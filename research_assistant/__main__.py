import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from .workflow import ResearchError, run


def main():
    parser = argparse.ArgumentParser(description='Research a topic through a LangGraph workflow.')
    parser.add_argument('topic')
    parser.add_argument('--mode', choices=['demo', 'extractive', 'ai'], default='demo')
    parser.add_argument('--sources', type=int, default=5, choices=range(1, 9), metavar='1-8')
    parser.add_argument('--output', type=Path, default=Path('reports/report.md'))
    args = parser.parse_args()
    load_dotenv()
    try:
        result = asyncio.run(run(args.topic, mode=args.mode, limit=args.sources))
    except (ResearchError, TimeoutError) as exc:
        parser.exit(1, f'Research failed: {exc}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result['report'], encoding='utf-8')
    print(f'Report written to {args.output}')


if __name__ == '__main__':
    main()
