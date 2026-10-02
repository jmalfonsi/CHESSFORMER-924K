"""The copied venv and launchers must resolve the mini project from any cwd."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_interpreter_loads_only_this_projects_source_from_another_directory(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run(
        [sys.executable, '-c',
         'import json, sys, chessformer; from chessformer.repo_env import ENV_PATH; '
         'from chessformer.mini import model; '
         'print(json.dumps([sys.prefix, chessformer.__file__, str(ENV_PATH), model.__file__]))'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    prefix, package, env_path, model = map(Path, json.loads(result.stdout))
    assert prefix == ROOT / '.venv'
    assert package == ROOT / 'src/chessformer/__init__.py'
    assert env_path == ROOT / '.env'
    assert model == ROOT / 'src/chessformer/mini/model.py'


def test_bot_launcher_resolves_its_own_directory_without_contacting_lichess(tmp_path):
    # A fake interpreter observes cwd and arguments without importing the bot
    # or opening a network connection.
    project = tmp_path / 'mini project'
    (project / 'tools').mkdir(parents=True)
    (project / '.venv/bin').mkdir(parents=True)
    launcher = project / 'tools/run_mini_bot.sh'
    shutil.copy2(ROOT / 'tools/run_mini_bot.sh', launcher)
    interpreter = project / '.venv/bin/python'
    interpreter.write_text('#!/bin/bash\nprintf "observed_cwd=%s\\n" "$PWD"\nprintf "arg=%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    result = subprocess.run(['bash', str(launcher)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert f'observed_cwd={project}' in result.stdout
    assert 'arg=chessformer.mini.lichess' in result.stdout
    assert 'arg=models/chessformer-924k-v1.pt' in result.stdout
    assert len(list((project / 'logs').glob('mini-bot-*.log'))) == 1


def test_ladder_launcher_uses_its_own_directory_and_allows_overrides(tmp_path):
    project = tmp_path / 'mini project'
    (project / 'tools').mkdir(parents=True)
    (project / '.venv/bin').mkdir(parents=True)
    launcher = project / 'tools/run_mini_ladder.sh'
    shutil.copy2(ROOT / 'tools/run_mini_ladder.sh', launcher)
    interpreter = project / '.venv/bin/python'
    interpreter.write_text('#!/bin/bash\nprintf "observed_cwd=%s\\n" "$PWD"\nprintf "arg=%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    result = subprocess.run(['bash', str(launcher), '--rounds', '4', '--account', 'MyOwnBot'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert f'observed_cwd={project}' in result.stdout
    assert result.stdout.count('arg=tools/challenge_ladder.py') == 1
    assert 'arg=--clock-limit\narg=180' in result.stdout
    assert 'arg=--clock-increment\narg=2' in result.stdout
    assert result.stdout.endswith('arg=--rounds\narg=4\narg=--account\narg=MyOwnBot\n')
