# Deploy EquiReach Mode A API to Hugging Face Spaces

1. Create a new Hugging Face Space named `equireach-api`.
2. Choose SDK: Docker, template: Blank, hardware: CPU basic, visibility: Public.
3. In the Space Files tab, keep the Space README front matter and make sure it includes:

```yaml
---
title: EquiReach API
sdk: docker
app_port: 7860
---
```

4. Upload only these files/folders from `apps/equireach` to the Space:
   - `Dockerfile`
   - `requirements.txt`
   - `equireach/`
   - `api/`

5. Do not upload `.venv`, `__pycache__`, `.pytest_cache`, private CSV files, `.sav` files, or anything from the repository root `data/` folder.

6. For testing, add Space variable `DEV=1`. This allows any site to call the API temporarily.

7. After the Space is running, test:
   - `https://YOUR_SPACE_URL/health`
   - `https://YOUR_SPACE_URL/api/demo?kind=selection&n=30&seed=1`

8. Before publishing the dashboard, remove `DEV=1` and set `ALLOWED_ORIGINS` to the exact Lovable site URL, with no trailing slash.
