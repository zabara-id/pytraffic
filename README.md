# pytraffic

## VS Code: настройка импортов для `src`-layout

Если проект лежит в `src/`, добавьте в репозиторий `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ],
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "${workspaceFolder}/src"
  }
}
```

И `.vscode/launch.json` для запуска/дебага:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File (PYTHONPATH=src)",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: pytraffic.optimize.problem",
      "type": "debugpy",
      "request": "launch",
      "module": "pytraffic.optimize.problem",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  ]
}
```
