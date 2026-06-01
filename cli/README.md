# vision-toolbelt CLI

A uv-installed CLI that turns images into standardized artifacts for coding agents.

Examples:

```bash
vision-toolbelt inspect image.png --out image.inspect.json
vision-toolbelt analyze image.png --prompt "loose connector, corrosion" --out image.analysis.json --overlay-out image.overlay.png
vision-toolbelt toolspec --format json
vision-toolbelt models list
```

The CLI is deliberately local-first. Heavy engines are optional and report structured `unavailable` warnings when dependencies or model files are not present.
