#!/bin/bash
set -e
export ANTHROPIC_API_KEY=$(cat /Users/GowthamSai/Documents/repos/gwthm-in/lossless-agent/benchmarks/longmemeval/.env | grep ANTHROPIC_API_KEY | cut -d= -f2)
export ANTHROPIC_BASE_URL=$(cat /Users/GowthamSai/Documents/repos/gwthm-in/lossless-agent/benchmarks/longmemeval/.env | grep ANTHROPIC_BASE_URL | cut -d= -f2)
cd /Users/GowthamSai/Documents/repos/gwthm-in/lossless-agent
exec /Users/GowthamSai/Documents/repos/gwthm-in/lossless-agent/.venv/bin/python /Users/GowthamSai/Documents/repos/gwthm-in/lossless-agent/benchmarks/longmemeval/run_full_benchmark.py 2>&1
