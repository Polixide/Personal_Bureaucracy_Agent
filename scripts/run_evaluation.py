"""Run sanitized V2 outcome/trajectory/reliability evaluations offline by default."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.scenarios import evaluate_scenarios, GOLDEN_SCENARIOS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory",action="store_true",help="Also run Neda memory/restart/accepted-repair harness offline")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--scenario",action="append",choices=list(GOLDEN_SCENARIOS))
    parser.add_argument("--live-model",action="store_true",help="Explicitly repeat the freeform proposal case with configured local Ollama; never external mutations")
    parser.add_argument("--judge-rag",action="store_true",help="Explicit independent configured Ollama RAG-triad judge; never used in production chat")
    parser.add_argument('--copilot-benchmark',action='store_true',help='Explicit live local assistant/planner benchmark on ten synthetic questions')
    parser.add_argument('--output',type=Path,help='Optional ignored local JSON output')
    args=parser.parse_args()
    if args.copilot_benchmark:
        import httpx
        from config import OLLAMA_HOST,LOCAL_BENCHMARK_MODELS,PROJECT_ROOT
        from src.evaluation.scenarios import benchmark_copilot
        response=httpx.get(OLLAMA_HOST+'/api/tags',timeout=10);response.raise_for_status()
        installed={row['name'] for row in response.json().get('models',[])}
        models=[name for name in LOCAL_BENCHMARK_MODELS if name in installed]
        if not models:raise SystemExit('No requested local benchmark model is installed; no models were downloaded')
        result=benchmark_copilot(models,directory=PROJECT_ROOT/'.local/copilot_stabilization')
        result['not_installed']=[name for name in LOCAL_BENCHMARK_MODELS if name not in installed]
        target=args.output or PROJECT_ROOT/'.local/copilot_stabilization/copilot_benchmark.json'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Live synthetic benchmark saved:',target)
        return

    model=None
    if args.live_model:
        from src.agents.assistant import ollama_tool_proposal
        model=ollama_tool_proposal
    judge=None
    if args.judge_rag:
        from src.evaluation.rag_eval import ollama_rag_judge
        judge=ollama_rag_judge
    result=evaluate_scenarios(repeats=args.repeats,names=args.scenario,assistant_model=model,rag_judge=judge)
    if args.memory:
        from src.evaluation.harness import evaluate
        memory=evaluate()
        result["memory_harness"]={key:value for key,value in memory.items() if key!="results"}
        if memory["scenarios_passed"]!=memory["scenarios_total"]:raise SystemExit("Memory evaluation failed")
    print(json.dumps(result,indent=2))
    if result["passed_runs"]!=result["run_count"]:raise SystemExit(1)

if __name__=="__main__":main()
